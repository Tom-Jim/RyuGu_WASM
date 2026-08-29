
fn taylor_remainder_bound(epsilon_max: f64, taylor_order: u32) -> Option<f64> {
    if !epsilon_max.is_finite() || epsilon_max >= 1.0 {
        return None;
    }
    let next_term = epsilon_max.powi((taylor_order + 1) as i32);
    let bound = next_term / (1.0 - epsilon_max).max(f64::EPSILON);
    bound.is_finite().then_some(bound)
}

fn taylor_gradient_remainder_bound(
    epsilon_max: f64,
    taylor_order: u32,
) -> Option<f64> {
    if !epsilon_max.is_finite() || epsilon_max >= 1.0 {
        return None;
    }
    let denominator = (1.0 - epsilon_max).max(f64::EPSILON).powi(2);
    let bound = f64::from(taylor_order + 1)
        * epsilon_max.powi(taylor_order as i32)
        / denominator;
    bound.is_finite().then_some(bound)
}
