//! Model geometry and density quadrature, independent of every gravity evaluator.

use crate::interface::components::*;
use bevy::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

pub(crate) const RADIAL_LAYER_COUNT: u32 = 4;
/// Live Verlet, invert reference, and Worker configure share this angular cap.
/// Full-mesh direct `source_sets` (tens of thousands of cells × 57 trees) is
/// what left FMM/FFT invert on "Preparing inversion observations…" forever.
pub(crate) const MAX_LIVE_ANGULAR_CELLS: usize = 32;

/// Builds the angular-cell/radial-layer discretization used by the radial model.
/// The mesh is assumed star-shaped with respect to its model origin, which is
/// true for the Ryugu asset. A non-star-shaped mesh would require multiple
/// radial intervals per angular cell.
pub fn build_density_quadrature_system(
    mut commands: Commands,
    topology: Option<Res<AsteroidTopologyGpuData>>,
    ryugu: Query<&Transform, With<RyuguMarker>>,
    existing: Option<Res<DensityQuadratureSource>>,
) {
    if existing.is_some() {
        return;
    }
    let Some(topology) = topology else { return };
    let Ok(transform) = ryugu.single() else {
        return;
    };
    if topology.triangles.is_empty() {
        return;
    }

    let scale = transform.scale.x;
    let mut cells = Vec::with_capacity(topology.triangles.len() / 3);
    let mut density_integral = 0.0_f64;
    let mut volume_integral = 0.0_f64;
    let mut solid_angle_sum = 0.0_f64;

    for tri in topology.triangles.as_chunks::<3>().0 {
        let p0 = topology.positions[tri[0] as usize] * scale;
        let p1 = topology.positions[tri[1] as usize] * scale;
        let p2 = topology.positions[tri[2] as usize] * scale;
        let Some((direction, radius, solid_angle)) = angular_cell(p0, p1, p2) else {
            continue;
        };
        cells.push((direction, radius, solid_angle));
        solid_angle_sum += solid_angle as f64;
        density_integral += solid_angle as f64
            * radial_density_integral(0.0, radius as f64, DENSITY_EPSILON as f64);
        volume_integral += solid_angle as f64 * (radius as f64).powi(3) / 3.0;
    }

    if cells.is_empty() || density_integral <= f64::EPSILON {
        error!("[gravity] failed to build radial-model angular cells");
        return;
    }

    let density_c = (RYUGU_MASS as f64 / density_integral) as f32;
    let constant_density = (RYUGU_MASS as f64 / volume_integral.max(f64::MIN_POSITIVE)) as f32;
    let mut bytes = Vec::with_capacity(cells.len() * RADIAL_LAYER_COUNT as usize * 32);
    let mut constant_bytes = Vec::with_capacity(bytes.capacity());

    for (direction, radius, solid_angle) in cells {
        for layer in 0..RADIAL_LAYER_COUNT {
            // Equal-volume shells avoid over-resolving the small central region.
            let inner_fraction = (layer as f32 / RADIAL_LAYER_COUNT as f32).cbrt();
            let outer_fraction = ((layer + 1) as f32 / RADIAL_LAYER_COUNT as f32).cbrt();
            let r_inner = radius * inner_fraction;
            let r_outer = radius * outer_fraction;
            let shell_measure = (r_outer.powi(3) - r_inner.powi(3)) / 3.0;
            let shell_integral =
                radial_density_integral(r_inner as f64, r_outer as f64, DENSITY_EPSILON as f64)
                    as f32;
            let density = density_c * shell_integral / shell_measure.max(f32::MIN_POSITIVE);

            push_f32s(
                &mut bytes,
                [direction.x, direction.y, direction.z, solid_angle],
            );
            push_f32s(&mut bytes, [r_inner, r_outer, density, 0.0]);
            push_f32s(
                &mut constant_bytes,
                [direction.x, direction.y, direction.z, solid_angle],
            );
            push_f32s(
                &mut constant_bytes,
                [r_inner, r_outer, constant_density, 0.0],
            );
        }
    }

    let count = (bytes.len() / 32) as u32;
    let source_hash = hash_bytes(&bytes);
    let constant_hash = hash_bytes(&constant_bytes);
    info!(
        "[gravity] radial model: {} angular cells, {} radial layers, solid-angle sum={:.6}, C={:.6e}",
        count / RADIAL_LAYER_COUNT,
        count,
        solid_angle_sum,
        density_c
    );
    if (solid_angle_sum - std::f64::consts::TAU * 2.0).abs() > 0.05 {
        warn!(
            "[gravity] mesh subtends {:.6} sr instead of 4pi; check that it is closed and star-shaped",
            solid_angle_sum
        );
    }

    commands.insert_resource(DensityC(density_c));
    commands.insert_resource(DensityQuadratureSource {
        bytes,
        constant_bytes,
        radius: cells_radius(&topology, scale),
        source_hash,
        constant_hash,
    });
}

/// Downsample angular cells the same way `configure_backend` does, scaling
/// each kept cell's solid angle by the number of skipped neighbours.
pub(crate) fn reduce_live_quadrature_bytes(bytes: &[u8]) -> Vec<u8> {
    const RECORD: usize = 32;
    let layers = RADIAL_LAYER_COUNT as usize;
    let count = bytes.len() / RECORD;
    if count <= MAX_LIVE_ANGULAR_CELLS * layers {
        return bytes.to_vec();
    }
    let angular = count / layers;
    let target = MAX_LIVE_ANGULAR_CELLS.min(angular);
    let stride = angular.div_ceil(target);
    let mut reduced = Vec::with_capacity(target * layers * RECORD);
    for angular_index in (0..angular).step_by(stride) {
        let represented = (angular - angular_index).min(stride) as f32;
        for layer in 0..layers {
            let offset = (angular_index * layers + layer) * RECORD;
            reduced.extend_from_slice(&bytes[offset..offset + RECORD]);
            let solid_at = reduced.len() - RECORD + 12;
            let solid = f32::from_le_bytes(reduced[solid_at..solid_at + 4].try_into().unwrap());
            reduced[solid_at..solid_at + 4].copy_from_slice(&(solid * represented).to_le_bytes());
        }
    }
    reduced
}

/// f64 cell records `[direction.xyz, solid, r_inner, r_outer, density, _]`.
pub(crate) fn reduce_live_cells(cells: &[f64]) -> Vec<f64> {
    let shells = cells.as_chunks::<8>().0;
    if shells.len() <= MAX_LIVE_ANGULAR_CELLS * RADIAL_LAYER_COUNT as usize {
        return cells.to_vec();
    }
    let layers = RADIAL_LAYER_COUNT as usize;
    let angular = shells.len() / layers;
    let target = MAX_LIVE_ANGULAR_CELLS.min(angular);
    let stride = angular.div_ceil(target);
    let mut result = Vec::with_capacity(target * layers * 8);
    for angular_index in (0..angular).step_by(stride) {
        for layer in 0..layers {
            let mut shell = shells[angular_index * layers + layer];
            let represented = (angular - angular_index).min(stride);
            shell[3] *= represented as f64;
            result.extend(shell);
        }
    }
    result
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(bytes);
    hasher.finish()
}

fn cells_radius(topology: &AsteroidTopologyGpuData, scale: f32) -> f32 {
    topology
        .positions
        .iter()
        .map(|position| position.length() * scale)
        .fold(0.0, f32::max)
        .max(f32::MIN_POSITIVE)
}

fn angular_cell(p0: Vec3, p1: Vec3, p2: Vec3) -> Option<(Vec3, f32, f32)> {
    let n0 = p0.try_normalize()?;
    let n1 = p1.try_normalize()?;
    let n2 = p2.try_normalize()?;
    let direction = (n0 + n1 + n2).try_normalize()?;

    let numerator = n0.dot(n1.cross(n2)).abs();
    let denominator = 1.0 + n0.dot(n1) + n1.dot(n2) + n2.dot(n0);
    let solid_angle = 2.0 * numerator.atan2(denominator);
    if !solid_angle.is_finite() || solid_angle <= 0.0 {
        return None;
    }

    let face_normal = (p1 - p0).cross(p2 - p0);
    let divisor = face_normal.dot(direction);
    let plane_radius = face_normal.dot(p0) / divisor;
    let centroid_radius = ((p0 + p1 + p2) / 3.0).length();
    let radius = if plane_radius.is_finite() && plane_radius > 0.0 {
        plane_radius
    } else {
        centroid_radius
    };
    (radius > 0.0).then_some((direction, radius, solid_angle))
}

/// Integral of `r^2 ln(1+r/epsilon)`, evaluated in f64.  This primitive makes
/// every radial layer exactly mass preserving for the shared logarithmic law.
fn radial_density_integral(inner: f64, outer: f64, epsilon: f64) -> f64 {
    fn primitive(r: f64, epsilon: f64) -> f64 {
        let logarithm = (1.0 + r / epsilon).ln();
        (r.powi(3) + epsilon.powi(3)) * logarithm / 3.0 - r.powi(3) / 9.0 + epsilon * r * r / 6.0
            - epsilon * epsilon * r / 3.0
    }
    primitive(outer, epsilon) - primitive(inner, epsilon)
}

fn push_f32s(bytes: &mut Vec<u8>, values: [f32; 4]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

#[cfg(test)]
mod live_reduction_tests {
    use super::{
        MAX_LIVE_ANGULAR_CELLS, RADIAL_LAYER_COUNT, reduce_live_cells, reduce_live_quadrature_bytes,
    };

    #[test]
    fn small_live_meshes_are_unchanged() {
        let cells: Vec<f64> = (0..MAX_LIVE_ANGULAR_CELLS * RADIAL_LAYER_COUNT as usize)
            .flat_map(|i| [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, i as f64, 0.0])
            .collect();
        assert_eq!(reduce_live_cells(&cells), cells);
    }

    #[test]
    fn large_meshes_keep_thirty_two_angular_cells() {
        let angular = 256;
        let layers = RADIAL_LAYER_COUNT as usize;
        let mut bytes = Vec::new();
        for a in 0..angular {
            for layer in 0..layers {
                for value in [
                    0.0_f32,
                    0.0,
                    1.0,
                    0.01,
                    layer as f32,
                    layer as f32 + 1.0,
                    a as f32,
                    0.0,
                ] {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        let reduced = reduce_live_quadrature_bytes(&bytes);
        assert_eq!(reduced.len() / 32, MAX_LIVE_ANGULAR_CELLS * layers);
        let first_solid = f32::from_le_bytes(reduced[12..16].try_into().unwrap());
        assert!((first_solid - 0.01 * (256 / 32) as f32).abs() < 1e-6);
    }
}
