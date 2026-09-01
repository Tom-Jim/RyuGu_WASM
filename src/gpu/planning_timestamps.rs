//! Optional pass-boundary timestamps. One device-lifetime allocation is shared
//! by every planning backend; only its current lease may encode or resolve it.
//! A lease stays alive through map/decode (or cancellation's queue fence).
//! Never substitute CPU wall time for a missing GPU measurement.
use bevy::prelude::*;
use bevy::render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder},
    renderer::{RenderDevice, RenderQueue},
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;
use wgpu29::{
    ComputePassTimestampWrites, ErrorFilter, Features, QuerySet, QuerySetDescriptor, QueryType,
};

// FFT batches at most eight columns of nine passes; Eq.106 needs four,
// FMM/evaluation one. Capacity never grows with source/RHS counts or repeats.
pub(crate) const PLANNING_TIMESTAMP_MAX_PASSES: u32 = 72;

#[derive(Resource, Default)]
pub(crate) struct PlanningTimestampPool {
    state: Arc<Mutex<TimestampPoolState>>,
}

#[derive(Default)]
enum TimestampPoolState {
    #[default]
    Uninitialized,
    Initializing,
    Ready(Arc<TimestampAllocation>),
    Disabled,
}

struct TimestampAllocation {
    query_set: QuerySet,
    resolve: Buffer,
    period_ns: f64,
    leased: AtomicBool,
}

impl Drop for TimestampAllocation {
    fn drop(&mut self) {
        // wgpu 29 has no public QuerySet::destroy; on WebGPU its native
        // release can wait for JS GC. Keep just one for the device lifetime,
        // and explicitly release the companion buffer at teardown.
        self.resolve.destroy();
    }
}

impl PlanningTimestampPool {
    /// Pending means yield without changing the job or submitting any passes.
    /// Ready(None) means timing is unavailable; GPU computation can continue.
    pub(crate) fn acquire(
        &self,
        device: &RenderDevice,
        queue: &RenderQueue,
        passes: u32,
    ) -> Poll<Option<PlanningTimestampQueries>> {
        assert!((1..=PLANNING_TIMESTAMP_MAX_PASSES).contains(&passes));
        let Ok(mut state) = self.state.try_lock() else {
            return Poll::Pending;
        };
        match &*state {
            TimestampPoolState::Ready(allocation) => {
                if allocation
                    .leased
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    return Poll::Pending;
                }
                return Poll::Ready(Some(PlanningTimestampQueries {
                    allocation: Arc::clone(allocation),
                    passes,
                }));
            }
            TimestampPoolState::Disabled => return Poll::Ready(None),
            TimestampPoolState::Initializing => return Poll::Pending,
            TimestampPoolState::Uninitialized => {}
        }
        let period_ns = f64::from(queue.get_timestamp_period());
        if !device.features().contains(Features::TIMESTAMP_QUERY)
            || !period_ns.is_finite()
            || period_ns <= 0.0
        {
            *state = TimestampPoolState::Disabled;
            return Poll::Ready(None);
        }
        *state = TimestampPoolState::Initializing;
        drop(state);

        // Feature support does not guarantee Metal can allocate a counter
        // sample buffer. Scope just this optional allocation and wait for all
        // results before allowing it into timestampWrites. Pop synchronously
        // so unrelated render work cannot be swallowed by these error scopes.
        let raw_device = device.wgpu_device();
        let oom_scope = raw_device.push_error_scope(ErrorFilter::OutOfMemory);
        let internal_scope = raw_device.push_error_scope(ErrorFilter::Internal);
        let validation_scope = raw_device.push_error_scope(ErrorFilter::Validation);
        let query_set = raw_device.create_query_set(&QuerySetDescriptor {
            label: Some("planning_shared_timestamps"),
            ty: QueryType::Timestamp,
            count: PLANNING_TIMESTAMP_MAX_PASSES * 2,
        });
        let resolve = device.create_buffer(&BufferDescriptor {
            label: Some("planning_shared_timestamp_resolve"),
            size: u64::from(PLANNING_TIMESTAMP_MAX_PASSES) * 16,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let validation_result = validation_scope.pop();
        let internal_result = internal_scope.pop();
        let oom_result = oom_scope.pop();
        let destination = Arc::clone(&self.state);
        bevy::tasks::AsyncComputeTaskPool::get().spawn(async move {
            let validation = validation_result.await;
            let internal = internal_result.await;
            let oom = oom_result.await;
            let mut state = destination.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(error) = oom.or(internal).or(validation) {
                // Do not retry every frame or put an invalid QuerySet in a
                // command buffer. Timing stays unavailable for this device.
                warn!("GPU timestamps unavailable: {error}. GPU calculation and pipeline totals remain enabled; no wall-time substitution.");
                *state = TimestampPoolState::Disabled;
            } else {
                *state = TimestampPoolState::Ready(Arc::new(TimestampAllocation {
                    query_set, resolve, period_ns, leased: AtomicBool::new(false),
                }));
            }
        }).detach();
        Poll::Pending
    }
}

pub(crate) struct PlanningTimestampQueries {
    allocation: Arc<TimestampAllocation>,
    passes: u32,
}

impl Drop for PlanningTimestampQueries {
    fn drop(&mut self) {
        self.allocation.leased.store(false, Ordering::Release);
    }
}

impl PlanningTimestampQueries {
    pub(crate) fn set_pass_count(&mut self, passes: u32) {
        assert!((1..=PLANNING_TIMESTAMP_MAX_PASSES).contains(&passes));
        self.passes = passes;
    }

    pub(crate) fn writes(&self, pass: u32) -> ComputePassTimestampWrites<'_> {
        assert!(pass < self.passes);
        ComputePassTimestampWrites {
            query_set: &self.allocation.query_set,
            beginning_of_pass_write_index: Some(pass * 2),
            end_of_pass_write_index: Some(pass * 2 + 1),
        }
    }

    pub(crate) fn resolve_into(&self, encoder: &mut CommandEncoder, staging: &Buffer, offset: u64) {
        encoder.resolve_query_set(
            &self.allocation.query_set,
            0..self.passes * 2,
            &self.allocation.resolve,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.allocation.resolve,
            0,
            staging,
            offset,
            u64::from(self.passes) * 16,
        );
    }

    pub(crate) fn decode(&self, bytes: &[u8]) -> Option<Vec<f64>> {
        let bytes = bytes.get(..self.passes as usize * 16)?;
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|pair| {
                let begin = u64::from_le_bytes(pair[..8].try_into().ok()?);
                let end = u64::from_le_bytes(pair[8..].try_into().ok()?);
                let ms = end.checked_sub(begin)? as f64 * self.allocation.period_ns / 1.0e6;
                // Zero is possible with browser timestamp quantization. Keep it
                // as measured zero, never invent an epsilon or a wall-time value.
                ms.is_finite().then_some(ms)
            })
            .collect()
    }
}
