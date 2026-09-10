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

#[derive(Deserialize)]
struct BatchRequest {
    positions: Vec<DVec3>,
    velocities: Vec<DVec3>,
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

/// Propagate a bounded time slice while batching every candidate target at
/// each integration endpoint. The browser ExaFMM build is single-threaded;
/// batching amortizes tree setup but does not enable OpenMP or GPU execution.
pub fn propagate_candidates(data: &str) -> Result<Vec<f64>, JsValue> {
    let request: BatchRequest =
        serde_json::from_str(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if request.positions.is_empty()
        || request.positions.len() > 2_048
        || request.sources.is_empty()
        || request.positions.len() != request.velocities.len()
        || request.velocities.iter().any(|v| !v.is_finite())
        || request.positions.iter().any(|p| !p.is_finite())
        || request.jets.is_empty()
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
        return Err("Invalid candidate batch request".into());
    }

    let source_xyz: Vec<f64> = request
        .sources
        .iter()
        .flat_map(|(p, _)| p.to_array())
        .collect();
    let source_masses: Vec<f64> = request.sources.iter().map(|(_, m)| *m).collect();
    let candidate_count = request.positions.len();
    let sample_count = request.jets.len();
    let mut positions = request.positions;
    let mut velocities = request.velocities;
    let mut output = vec![0.0; candidate_count * sample_count * 6];
    let mut targets = vec![0.0; candidate_count * 3];
    let mut accelerations = vec![DVec3::ZERO; candidate_count];

    // Position and time do not change between the endpoint kick and the next
    // step's initial kick. Reuse that exact FMM result, without extrapolation.
    for (sample, jet) in request.jets.iter().enumerate() {
        let dt = if sample == 0 {
            0.0
        } else {
            jet.time - request.jets[sample - 1].time
        };
        if sample > 0 {
            for candidate in 0..candidate_count {
                velocities[candidate] += accelerations[candidate] * (0.5 * dt);
                positions[candidate] += velocities[candidate] * dt;
            }
        }

        // A one-sample request only copies the initial state.
        if sample_count > 1 {
            let inverse_rotation = jet.rotation.inverse();
            for (target, position) in targets.chunks_exact_mut(3).zip(&positions) {
                target.copy_from_slice(&(inverse_rotation * *position).to_array());
            }
            let fields = super::field_sources("fmm", &source_xyz, &source_masses, &targets)?;
            if fields.len() != candidate_count * 4
                || fields.iter().any(|value| !value.is_finite())
            {
                return Err("Invalid batched FMM field response".into());
            }
            for (acceleration, field) in accelerations.iter_mut().zip(fields.chunks_exact(4)) {
                *acceleration = jet.rotation * DVec3::from_slice(&field[..3]);
            }
        }

        for candidate in 0..candidate_count {
            if sample > 0 {
                velocities[candidate] += accelerations[candidate] * (0.5 * dt);
            }
            if !positions[candidate].is_finite() || !velocities[candidate].is_finite() {
                return Err("Non-finite candidate trajectory".into());
            }
            let offset = (candidate * sample_count + sample) * 6;
            output[offset..offset + 3].copy_from_slice(&positions[candidate].to_array());
            output[offset + 3..offset + 6].copy_from_slice(&velocities[candidate].to_array());
        }
    }
    Ok(output)
}
