fn half_line_quadrature_bytes(length_scale: f32) -> Vec<u8> {
    let record_count = QUADRATURE_COUNT as usize * (FREQUENCY_COUNT as usize + 1);
    let mut records = Vec::with_capacity(record_count);
    let du = 1.0 / QUADRATURE_COUNT as f32;
    let length_scale = length_scale.max(1.0);
    for index in 0..QUADRATURE_COUNT {
        let u = (index as f32 + 0.5) * du;
        let denominator = 1.0 - u;
        // h = L u/(1-u), dh/du = L/(1-u)^2. Choosing L=1/sigma
        // resolves the physical Laplace-decay length instead of concentrating
        // nearly every node inside the first few metres.
        let h = length_scale * u / denominator;
        let weight = length_scale * du / (denominator * denominator);
        records.push([h, weight]);
    }
    // The same nonuniform Laplace phase was previously recomputed inside every
    // coefficient/voxel/segment invocation.  Cache its 129x64 matrix once;
    // ordinary FFT is not applicable because h=L*u/(1-u) is nonuniform.
    let sigma = length_scale.recip();
    for frequency_index in 0..FREQUENCY_COUNT {
        let signed_frequency = frequency_index as i32 - HALF_COUNT as i32;
        let omega = signed_frequency as f32 * 0.002;
        for quadrature_index in 0..QUADRATURE_COUNT as usize {
            let h = records[quadrature_index][0];
            let attenuation = (-sigma * h).exp();
            let angle = -omega * h;
            let (sine, cosine) = angle.sin_cos();
            records.push([attenuation * cosine, attenuation * sine]);
        }
    }
    bytemuck::cast_slice(&records).to_vec()
}

fn uniform_bytes(
    probe: Vec3,
    origin: Vec3,
    direction: Vec3,
    source_count: u32,
    radius: f32,
    longitudinal_limit: f32,
    taylor_order: u32,
    density_mode_count: u32,
    segment_id: u32,
    evaluate_dual_certificate: bool,
    // 0 = direct signed-spectrum, 1 = potential-only, 2 = Type-2 NUFFT.
    inversion_mode: u32,
    target_count: u32,
    target_offset: u32,
) -> [u8; 96] {
    let mut bytes = [0_u8; 96];
    for (offset, value) in [
        (0, probe.x),
        (4, probe.y),
        (8, probe.z),
        (12, G),
        (16, origin.x),
        (20, origin.y),
        (24, origin.z),
        (28, 2.0 / radius.max(1.0)),
        (32, direction.x),
        (36, direction.y),
        (40, direction.z),
        (44, 0.002),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (48, source_count),
        (52, HALF_COUNT),
        (56, QUADRATURE_COUNT),
        (60, taylor_order.clamp(1, TAYLOR_MAX_ORDER)),
        (64, density_mode_count),
        (68, segment_id),
        (72, u32::from(evaluate_dual_certificate)),
        (76, target_count.max(1)),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[80..84].copy_from_slice(&longitudinal_limit.max(1.0).to_le_bytes());
    bytes[84..88].copy_from_slice(&target_offset.to_le_bytes());
    bytes[88..92].copy_from_slice(&inversion_mode.to_le_bytes());
    bytes
}

fn uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    buffer_entry(binding, BufferBindingType::Uniform)
}

fn storage_ro_entry(binding: u32) -> BindGroupLayoutEntry {
    buffer_entry(binding, BufferBindingType::Storage { read_only: true })
}
fn storage_rw_entry(binding: u32) -> BindGroupLayoutEntry {
    buffer_entry(binding, BufferBindingType::Storage { read_only: false })
}
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
