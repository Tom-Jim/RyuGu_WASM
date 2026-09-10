/// Direct source sum retained as the independent certified reference.  It is
/// retained for method-independent accuracy checks: certification needs an
/// exact, method-independent field/gradient, while candidate dynamics use the
/// source-count-independent FMM tree and reevaluate every updated position.
#[cfg(test)]
pub(crate) fn evaluate_planning_reference_field(
    target: DVec3,
    basis_records: &[PlanningBasisRecord],
    densities: &[f32],
) -> Option<(DVec3, DMat3)> {
    let mut acceleration = DVec3::ZERO;
    let mut gradient = DMat3::ZERO;
    accumulate_planning_reference_chunk(
        target,
        basis_records,
        densities,
        &mut acceleration,
        &mut gradient,
    )?;
    Some((acceleration, gradient))
}

/// Independent f64 oracle. WebGPU has no portable shader-f64, so keep this
/// verification on CPU but bound each caller's work and retain sum order.
pub(crate) fn accumulate_planning_reference_chunk(
    target: DVec3,
    basis_records: &[PlanningBasisRecord],
    densities: &[f32],
    acceleration: &mut DVec3,
    gradient: &mut DMat3,
) -> Option<()> {
    if densities.len() != 56 || !target.is_finite() {
        return None;
    }
    let sources = basis_records
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
        .collect::<Option<Vec<_>>>()?;
    let h = 0.01;
    let mut targets = vec![target];
    for axis in [DVec3::X, DVec3::Y, DVec3::Z] {
        targets.extend([target + h * axis, target - h * axis]);
    }
    let values = crate::cpp_backend::evaluate_sources("direct", &sources, &targets).ok()?;
    let field = |i: usize| DVec3::new(values[i][0], values[i][1], values[i][2]);
    *acceleration += field(0);
    *gradient += DMat3::from_cols(
        (field(1) - field(2)) / (2.0 * h),
        (field(3) - field(4)) / (2.0 * h),
        (field(5) - field(6)) / (2.0 * h),
    );
    (acceleration.is_finite() && gradient.is_finite()).then_some(())
}
