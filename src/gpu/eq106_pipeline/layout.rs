fn half_line_quadrature_bytes(length_scale: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(QUADRATURE_COUNT as usize * 8);
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
        bytes.extend_from_slice(&h.to_le_bytes());
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    bytes
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
    inversion_mode: bool,
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
    for (offset, value) in [(80, longitudinal_limit.max(1.0))] {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes[84..88].copy_from_slice(&target_offset.to_le_bytes());
    bytes[88..92].copy_from_slice(&u32::from(inversion_mode).to_le_bytes());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    fn snapshot(request_id: u64, body_position: Vec3) -> GravityRequestSnapshot {
        GravityRequestSnapshot {
            request_id,
            epoch: 7,
            simulation_time_seconds: request_id as f64,
            body_position,
            ryugu_transform: Transform::IDENTITY,
            probe_position: body_position,
            probe_velocity: Vec3::X,
        }
    }

    fn output_rows(position: Vec3, acceleration: Vec3) -> [[f32; 4]; 9] {
        let mut rows = [[0.0; 4]; 9];
        rows[0] = [acceleration.x, acceleration.y, acceleration.z, 1.0];
        rows[6] = [1.0, 0.0, 0.0, 0.0];
        rows[7] = [position.x, position.y, position.z, 2.0];
        rows[8] = [10.0, 20.0, 30.0, 0.0];
        rows
    }

    #[test]
    fn taylor_orders_allocate_only_active_coefficients() {
        assert_eq!(taylor_coefficient_count(1), 3);
        assert_eq!(taylor_coefficient_count(2), 6);
        assert_eq!(taylor_coefficient_count(3), 10);
        assert_eq!(taylor_coefficient_count(4), 15);
    }

    #[test]
    fn large_target_batches_use_a_two_dimensional_dispatch() {
        assert_eq!(target_dispatch_grid(1), (1, 1));
        assert_eq!(target_dispatch_grid(65_535), (65_535, 1));
        assert_eq!(target_dispatch_grid(90_166), (65_535, 2));
    }

    #[test]
    fn uniform_layout_contains_target_count_and_active_order() {
        let bytes = uniform_bytes(
            Vec3::ONE,
            Vec3::ZERO,
            Vec3::X,
            123,
            464.765,
            900.0,
            1,
            544,
            9,
            false,
            false,
            90_166,
            0,
        );

        assert_eq!(bytes.len(), 96);
        assert_eq!(read_u32(&bytes, 48), 123);
        assert_eq!(read_u32(&bytes, 60), 1);
        assert_eq!(read_u32(&bytes, 64), 544);
        assert_eq!(read_u32(&bytes, 68), 9);
        assert_eq!(read_u32(&bytes, 72), 0);
        assert_eq!(read_u32(&bytes, 76), 90_166);
    }

    #[test]
    fn batch_decoder_preserves_every_target_block() {
        let positions = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
        let accelerations = [Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.4, 0.5, 0.6)];
        let snapshots = positions
            .iter()
            .enumerate()
            .map(|(index, position)| snapshot(index as u64, *position))
            .collect::<Vec<_>>();
        let partial_sums = positions
            .iter()
            .zip(accelerations)
            .flat_map(|(position, acceleration)| output_rows(*position, acceleration))
            .collect();
        let packet = Eq106ReadbackPacket {
            partial_sums,
            snapshots,
            batch_capture_id: Some(42),
            sensitivity_column_count: 0,
            sensitivity_source_hash: 0,
            sensitivity_basis_hash: 0,
            sensitivity_configuration_hash: 0,
            timings: default(),
        };

        let decoded = decode_eq106_packet(&packet, Vec3::ZERO).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].snapshot.request_id, 0);
        assert_eq!(decoded[1].snapshot.request_id, 1);
        assert_eq!(decoded[0].body_acceleration, accelerations[0]);
        assert_eq!(decoded[1].body_acceleration, accelerations[1]);
    }

    #[test]
    fn adaptive_batch_elements_cover_the_trajectory_without_gaps() {
        let radius = 620.0_f32;
        let positions = (0..181)
            .map(|index| {
                let angle = 0.35 * index as f32 / 180.0;
                Vec3::new(-radius * angle.cos(), radius * angle.sin(), -65.459)
            })
            .collect::<Vec<_>>();
        let velocities = positions
            .iter()
            .map(|position| Vec3::new(-position.y, position.x, 0.0).normalize() * 0.235_503)
            .collect::<Vec<_>>();

        let elements = build_trajectory_batch_elements(&positions, &velocities, 464.765, 1_200.0);
        assert!(!elements.is_empty());
        assert_eq!(elements[0].target_offset, 0);
        assert_eq!(
            elements
                .iter()
                .map(|element| element.target_count as usize)
                .sum::<usize>(),
            positions.len()
        );
        for pair in elements.windows(2) {
            assert_eq!(
                pair[0].target_offset + pair[0].target_count,
                pair[1].target_offset
            );
        }
        assert!(elements.iter().all(|element| {
            (1..=TAYLOR_MAX_ORDER).contains(&element.taylor_order)
                && element.line_limit >= 0.35 * 464.765
                && element.line_limit <= 1_200.0
        }));
    }
}
