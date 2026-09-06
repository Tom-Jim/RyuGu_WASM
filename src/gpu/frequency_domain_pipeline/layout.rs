fn reciprocal_space_quadrature_bytes(source_radius: f32) -> Vec<u8> {
    let mut records = Vec::with_capacity(QUADRATURE_COUNT as usize);
    for index in 0..QUADRATURE_COUNT {
        let (wave_vector, volume_weight) = eq184_quadrature_node(
            index as usize,
            f64::from(source_radius),
        )
        .expect("equation-(184) quadrature index and source radius are validated");
        records.push([
            wave_vector.x as f32,
            wave_vector.y as f32,
            wave_vector.z as f32,
            volume_weight as f32,
        ]);
    }
    bytemuck::cast_slice(&records).to_vec()
}

fn uniform_bytes(
    origin: Vec3,
    source_count: u32,
    spectrum_slot: u32,
    // 0 = full field, 1 = compact sensitivity column.
    inversion_mode: u32,
    target_count: u32,
    target_offset: u32,
    source_layout: u32,
    input_base: u32,
) -> [u8; 52] {
    // All members of FrequencyDomainParams are scalar f32/u32 values. This
    // deliberately avoids vec3 alignment differences across WebGPU backends;
    // the resulting storage-array stride is exactly 52 bytes.
    let mut bytes = [0_u8; 52];
    for (offset, value) in [
        (0, G),
        (36, origin.x),
        (40, origin.y),
        (44, origin.z),
        // A small positive s keeps exp(-s t) finite over the captured
        // trajectory while preserving the required Re(s)>0 condition.
        (48, EQ184_BASE_LAPLACE_SIGMA as f32),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (offset, value) in [
        (4, source_count),
        (8, QUADRATURE_COUNT),
        (12, spectrum_slot),
        (16, target_count.max(1)),
        (20, target_offset),
        (24, inversion_mode),
        (28, source_layout),
        (32, input_base),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
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
