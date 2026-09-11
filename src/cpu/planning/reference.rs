/// Independent f64 oracle. WebGPU has no portable shader-f64, so this
/// verification runs as direct sums in the numerical Worker. The Worker
/// evaluates the sources in consecutive chunks of this size and answers one
/// field block per chunk; the blocks are accumulated here in chunk order.
/// Chunking only existed to time-slice the former main-thread loop; on the
/// Worker a single call per request is preferred, so the chunk is large enough
/// to cover the default First/Stress source count in one evaluate_sources.
pub(crate) const PLANNING_REFERENCE_SOURCE_CHUNK: u32 = 65_536;
/// Targets per reference point: the point plus `±h` along each axis.
pub(crate) const PLANNING_REFERENCE_STENCIL: usize = 7;
const PLANNING_REFERENCE_STEP: f64 = 0.01;

/// Density-weighted point sources of one density model row.
pub(crate) fn planning_reference_sources(
    basis_records: &[PlanningBasisRecord],
    densities: &[f32],
) -> Option<Vec<(DVec3, f64)>> {
    if densities.len() != 56 {
        return None;
    }
    basis_records
        .iter()
        .map(|record| {
            Some((
                DVec3::new(
                    record.position_volume[0] as f64,
                    record.position_volume[1] as f64,
                    record.position_volume[2] as f64,
                ),
                record.position_volume[3] as f64
                    * f64::from(*densities.get(record.voxel_index as usize)?),
            ))
        })
        .collect()
}

/// Appends the seven stencil targets of one finite reference point.
pub(crate) fn push_planning_reference_stencil(target: DVec3, targets: &mut Vec<DVec3>) -> bool {
    if !target.is_finite() {
        return false;
    }
    targets.push(target);
    for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
        targets.extend([
            target + PLANNING_REFERENCE_STEP * axis,
            target - PLANNING_REFERENCE_STEP * axis,
        ]);
    }
    true
}

/// Accumulates the per-chunk stencil fields of one reference point.
///
/// `chunks` yields the seven `[gx, gy, gz, potential]` records of this point
/// for every source chunk, in chunk order. Returns `None` when the sum stops
/// being finite, matching the former incremental accumulation.
pub(crate) fn accumulate_planning_reference<'a>(
    chunks: impl Iterator<Item = &'a [[f64; 4]]>,
) -> Option<(DVec3, DMat3)> {
    let h = PLANNING_REFERENCE_STEP;
    let mut acceleration = DVec3::ZERO;
    let mut gradient = DMat3::ZERO;
    for stencil in chunks {
        if stencil.len() != PLANNING_REFERENCE_STENCIL
            || stencil.iter().flatten().any(|value| !value.is_finite())
        {
            return None;
        }
        let field = |i: usize| DVec3::new(stencil[i][0], stencil[i][1], stencil[i][2]);
        acceleration += field(0);
        gradient += DMat3::from_cols(
            (field(1) - field(2)) / (2.0 * h),
            (field(3) - field(4)) / (2.0 * h),
            (field(5) - field(6)) / (2.0 * h),
        );
        if !(acceleration.is_finite() && gradient.is_finite()) {
            return None;
        }
    }
    Some((acceleration, gradient))
}
