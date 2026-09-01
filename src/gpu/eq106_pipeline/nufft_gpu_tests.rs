//! Opt-in native GPU regression: production FFT + production interpolation
//! versus an independent f64 Fourier sum, with no browser or benchmark timer.
use num_complex::Complex64;
use wgpu::util::DeviceExt;
use wgpu29 as wgpu;

const GRID: usize = 1024;
const HALF: usize = 64;
const PAIRS: usize = 6;
const SIGMA: f32 = 0.125;
const OMEGA: f32 = 0.75;
const COORDINATES: [f32; 8] = [0.0, 0.125, 1.5, 127.25, 511.75, 777.125, 1023.75, 1024.0];

fn selected(sample: [f32; 8], pair: usize, frequency: usize) -> [Complex64; 2] {
    let values: [Complex64; 4] = std::array::from_fn(|i| {
        Complex64::new(f64::from(sample[2 * i]), f64::from(sample[2 * i + 1]))
    });
    let derivative = Complex64::new(
        f64::from(SIGMA),
        (frequency as i32 - HALF as i32) as f64 * f64::from(OMEGA),
    );
    match pair {
        0 => [values[0], values[1]],
        1 => [values[2], values[3]],
        2 => [values[0] * derivative, values[1] * derivative],
        3 => [values[2] * derivative, Complex64::default()],
        4 if frequency % 2 == 0 => [values[0], values[1]],
        5 if frequency % 2 == 0 => [values[2], Complex64::default()],
        _ => [Complex64::default(); 2],
    }
}

#[test]
#[ignore = "explicit GPU validation; never run implicitly during a code-only repair"]
fn production_nufft_matches_direct_fourier_on_gpu() {
    bevy::tasks::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("GPU adapter required; no silent CPU substitution");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("GPU device");
        // Import the actual evaluator interpolation function, not a test copy.
        let evaluator = include_str!("../../wgsl/eq106_complex.wgsl");
        let start = evaluator.find("struct NufftInterpolation").unwrap();
        let end = evaluator[start..].find("\n#endif").unwrap() + start;
        let interpolation = evaluator[start..end].replace("line_samples[", "nufft_storage[");
        let shader_source = format!(
            "{}\n{}\n{}",
            include_str!("../../wgsl/eq106_nufft.wgsl"),
            interpolation,
            r#"
const NUFFT_GRID_SIZE: u32 = 1024u;
const NUFFT_PAIR_COUNT: u32 = 6u;
var<private> eq_params: Eq106Params;
@compute @workgroup_size(1)
fn sample_interpolation(@builtin(global_invocation_id) id: vec3<u32>) {
    eq_params = segment_params[0];
    let coordinates = array<f32, 8>(0.0, 0.125, 1.5, 127.25, 511.75, 777.125, 1023.75, 1024.0);
    let period = 2.0 * PI / eq_params.omega_step;
    let h = coordinates[id.x] / 1024.0 * period;
    nufft_storage[6144u + id.y * 8u + id.x] = type2_nufft_interpolate(0u, id.y, h).value;
}
"#
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production NUFFT regression"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let entries = [
            (0, false, false),
            (3, false, true),
            (6, false, true),
            (7, true, false),
        ]
        .map(|(binding, uniform, writable)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if uniform {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: !writable,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let fft_pipeline = pipeline("build_type2_nufft_grid");
        let interpolation_pipeline = pipeline("sample_interpolation");
        let mut params = [0u32; 24];
        params[7] = SIGMA.to_bits();
        params[11] = OMEGA.to_bits();
        params[13] = HALF as u32;
        params[17] = 1; // segment_id; Taylor order zero = one coefficient
        let parameter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let density_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &[0u8; 544 * 16],
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let output_size = ((GRID + COORDINATES.len()) * PAIRS * 16) as u64;
        // DC catches 1/1024 attenuation. Isolated positive/negative complex
        // modes catch sign/bin errors. The full band exercises all six pairs.
        for fixture in 0..4 {
            let spectrum: Vec<[f32; 8]> = (0..=2 * HALF)
                .map(|frequency| {
                    let occupied = match fixture {
                        0 => frequency == HALF,
                        1 => frequency == HALF + 31,
                        2 => frequency == HALF - 23,
                        _ => true,
                    };
                    std::array::from_fn(|channel| {
                        if occupied {
                            ((frequency * 13 + channel * 7 + 1) as f32 * 0.173).sin()
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            let spectrum_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&spectrum),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: output_size,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: output_size,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            });
            let bindings = [
                (0, &parameter_buffer),
                (3, &spectrum_buffer),
                (6, &output),
                (7, &density_buffer),
            ]
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &bindings,
            });
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            for (pipeline, x, y, z) in [
                (&fft_pipeline, 1, 1, PAIRS as u32),
                (
                    &interpolation_pipeline,
                    COORDINATES.len() as u32,
                    PAIRS as u32,
                    1,
                ),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x, y, z);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
            let submission = queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap();
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(std::time::Duration::from_secs(30)),
                })
                .unwrap();
            receiver
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap()
                .unwrap();
            let view = staging.slice(..).get_mapped_range();
            let values: Vec<f64> = view
                .chunks_exact(4)
                .map(|bytes| f64::from(f32::from_le_bytes(bytes.try_into().unwrap())))
                .collect();
            for pair in 0..PAIRS {
                for (point, coordinate) in (0..GRID).map(|i| (i, i as f64)).chain(
                    COORDINATES
                        .iter()
                        .enumerate()
                        .map(|(i, x)| (GRID + i, f64::from(*x))),
                ) {
                    let mut expected = [Complex64::default(); 2];
                    let mut scale = [0.0_f64; 2];
                    for (frequency, sample) in spectrum.iter().copied().enumerate() {
                        let phase = Complex64::from_polar(
                            1.0,
                            std::f64::consts::TAU
                                * (frequency as i32 - HALF as i32) as f64
                                * coordinate
                                / GRID as f64,
                        );
                        let modes = selected(sample, pair, frequency);
                        for channel in 0..2 {
                            expected[channel] += modes[channel] * phase;
                            scale[channel] += modes[channel].norm();
                        }
                    }
                    let offset = if point < GRID {
                        pair * GRID + point
                    } else {
                        PAIRS * GRID + pair * COORDINATES.len() + point - GRID
                    };
                    for channel in 0..2 {
                        let actual = Complex64::new(
                            values[offset * 4 + channel * 2],
                            values[offset * 4 + channel * 2 + 1],
                        );
                        let tolerance = if point < GRID { 3.0e-5 } else { 2.0e-3 };
                        assert!(
                            (actual - expected[channel]).norm()
                                <= tolerance * scale[channel].max(1.0e-6),
                            "fixture {fixture}, pair {pair}, point {point}, channel {channel}: {actual} != {}",
                            expected[channel]
                        );
                    }
                }
            }
            drop(view);
            staging.unmap();
        }
    });
}
