use glam::{DMat3, DQuat, DVec3};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[derive(Clone, Copy, Deserialize)]
struct Jet {
    time: f64,
    rotation: DQuat,
    position: DVec3,
    acceleration: DVec3,
    jacobian: DMat3,
}

#[derive(Deserialize)]
struct Request {
    position: DVec3,
    velocity: DVec3,
    jets: Vec<Jet>,
    sources: Vec<(DVec3, f64)>,
}

/// Candidate trials are independent of the authoritative live simulation clock.
#[wasm_bindgen]
pub fn propagate_candidate(data: &str) -> Result<Vec<f64>, JsValue> {
    let request: Request =
        serde_json::from_str(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if request.jets.is_empty()
        || !request.position.is_finite()
        || !request.velocity.is_finite()
        || request.jets.iter().any(|j| {
            !j.time.is_finite()
                || !j.rotation.is_finite()
                || (j.rotation.length_squared() - 1.0).abs() > 1e-5
                || !j.position.is_finite()
                || !j.acceleration.is_finite()
                || !j.jacobian.is_finite()
        })
        || request.jets.windows(2).any(|w| w[1].time <= w[0].time)
        || request
            .sources
            .iter()
            .any(|(p, m)| !p.is_finite() || !m.is_finite() || *m <= 0.0)
    {
        return Err("Invalid candidate trajectory request".into());
    }
    let xyz: Vec<_> = request
        .sources
        .iter()
        .flat_map(|(p, _)| p.to_array())
        .collect();
    let masses: Vec<_> = request.sources.iter().map(|(_, m)| *m).collect();
    let acceleration = |jet: &Jet, position: DVec3| -> Result<DVec3, JsValue> {
        let value = if masses.is_empty() {
            jet.acceleration + jet.jacobian * (position - jet.position)
        } else {
            let target = jet.rotation.inverse() * position;
            let field = super::field_sources("fmm", &xyz, &masses, &target.to_array())?;
            if field.len() != 4 {
                return Err("Invalid candidate field response".into());
            }
            jet.rotation * DVec3::from_slice(&field)
        };
        if !value.is_finite() {
            return Err("Non-finite candidate acceleration".into());
        }
        Ok(value)
    };
    let mut position = request.position;
    let mut velocity = request.velocity;
    let mut output = Vec::with_capacity(request.jets.len() * 6);
    for (index, jet) in request.jets.iter().enumerate() {
        if index > 0 {
            let previous = &request.jets[index - 1];
            let dt = jet.time - previous.time;
            velocity += acceleration(previous, position)? * (0.5 * dt);
            position += velocity * dt;
            velocity += acceleration(jet, position)? * (0.5 * dt);
        }
        if !position.is_finite() || !velocity.is_finite() {
            return Err("Non-finite candidate trajectory".into());
        }
        output.extend(position.to_array());
        output.extend(velocity.to_array());
    }
    Ok(output)
}
