use glam::Vec3;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

const RYUGU_MASS: f32 = 4.5e11;
const SMOOTHNESS_WEIGHT: f64 = 0.02;
const RADIAL_SYMMETRY_WEIGHT: f64 = 0.15;
const PRIOR_WEIGHT: f64 = 0.0001;
const OBSERVATION_NOISE_FRACTION: f32 = 1e-3;
const OBSERVATION_NOISE_FLOOR: f32 = 1e-12;

#[derive(Deserialize)]
struct Voxel {
    volume: f32,
    baseline_density: f32,
    center: Vec3,
}
#[derive(Deserialize)]
struct DensityProblem {
    voxels: Vec<Voxel>,
    observations: Vec<Vec3>,
    observed_accelerations: Vec<Vec3>,
    sensitivities: Vec<Vec3>,
    data_error_scale: f64,
    neighbours: Vec<(usize, usize)>,
    voxel_size: f32,
    homogeneous: bool,
}

#[wasm_bindgen]
pub fn solve_density(data: &str) -> Result<Vec<f32>, JsValue> {
    let problem: DensityProblem =
        serde_json::from_str(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if problem
        .neighbours
        .iter()
        .any(|&(a, b)| a >= problem.voxels.len() || b >= problem.voxels.len())
    {
        return Err("Invalid density neighbour index".into());
    }
    solve_density_qp(&problem, &problem.observations).map_err(|e| e.into())
}

fn solve_density_qp(job: &DensityProblem, observations: &[Vec3]) -> Result<Vec<f32>, String> {
    use clarabel::{algebra::CscMatrix, solver::*};
    let n = job.voxels.len();
    if n == 0
        || observations.is_empty()
        || observations.len() != job.observed_accelerations.len()
        || job.sensitivities.len() != n * observations.len()
    {
        return Err("Clarabel QP dimensions do not match the frozen observations.".into());
    }
    let scale = job.data_error_scale.max(1.0e-24);
    let mut h = vec![vec![0.0_f64; n]; n];
    let mut g = vec![0.0_f64; n];
    for (observation, observed) in observations.iter().enumerate() {
        let base = observation * n;
        let sigma = (job.observed_accelerations[observation].length() * OBSERVATION_NOISE_FRACTION)
            .max(OBSERVATION_NOISE_FLOOR);
        let weight = 1.0 / (observations.len().max(1) as f64 * f64::from(sigma * sigma) * scale);
        for i in 0..n {
            let si = job.sensitivities[base + i];
            g[i] += si.dot(*observed) as f64 * weight;
            for (j, sj) in job.sensitivities[base..base + i + 1].iter().enumerate() {
                h[j][i] += si.dot(*sj) as f64 * weight;
            }
        }
    }
    let mean_density = (RYUGU_MASS as f64
        / job
            .voxels
            .iter()
            .map(|voxel| voxel.volume as f64)
            .sum::<f64>())
    .max(f64::MIN_POSITIVE);
    for i in 0..n {
        let baseline = job.voxels[i].baseline_density.max(f32::MIN_POSITIVE) as f64;
        let prior = PRIOR_WEIGHT / (n as f64 * baseline * baseline);
        h[i][i] += prior + 1.0e-12;
        g[i] += prior * baseline;
    }
    for &(left, right) in &job.neighbours {
        add_pair_penalty(
            &mut h,
            left,
            right,
            SMOOTHNESS_WEIGHT / (job.neighbours.len().max(1) as f64 * mean_density * mean_density),
        );
    }
    let radial_pairs = radial_density_pairs(job);
    for &(left, right) in &radial_pairs {
        add_pair_penalty(
            &mut h,
            left,
            right,
            RADIAL_SYMMETRY_WEIGHT
                / (radial_pairs.len().max(1) as f64 * mean_density * mean_density),
        );
    }
    let mut p_rows = Vec::new();
    let mut p_cols = Vec::new();
    let mut p_vals = Vec::new();
    for (row, h_row) in h.iter().enumerate() {
        for (col, value) in h_row.iter().enumerate().skip(row) {
            if value.abs() > 0.0 {
                p_rows.push(row);
                p_cols.push(col);
                p_vals.push(2.0 * value);
            }
        }
    }
    let p = CscMatrix::new_from_triplets(n, n, p_rows, p_cols, p_vals);
    let homogeneous_equalities = if job.homogeneous {
        n.saturating_sub(1)
    } else {
        0
    };
    let rows = 2 * n + 1 + homogeneous_equalities;
    let nonzeros = 3 * n + 2 * homogeneous_equalities;
    let mut a_rows = Vec::with_capacity(nonzeros);
    let mut a_cols = Vec::with_capacity(nonzeros);
    let mut a_vals = Vec::with_capacity(nonzeros);
    let mut b = vec![0.0_f64; rows];
    for col in 0..n {
        a_rows.extend([col, n + col, 2 * n]);
        a_cols.extend([col, col, col]);
        a_vals.extend([1.0, -1.0, job.voxels[col].volume as f64]);
        b[col] = 8.0 * job.voxels[col].baseline_density as f64;
        b[n + col] = -0.02 * job.voxels[col].baseline_density as f64;
    }
    b[2 * n] = RYUGU_MASS as f64;
    for index in 1..=homogeneous_equalities {
        let row = 2 * n + index;
        a_rows.extend([row, row]);
        a_cols.extend([0, index]);
        a_vals.extend([-1.0, 1.0]);
    }
    let a = CscMatrix::new_from_triplets(rows, n, a_rows, a_cols, a_vals);
    let cones = [
        NonnegativeConeT(2 * n),
        ZeroConeT(1 + homogeneous_equalities),
    ];
    let settings = DefaultSettings {
        verbose: false,
        max_iter: 500,
        ..Default::default()
    };
    let q = g.iter().map(|value| -2.0 * value).collect::<Vec<_>>();
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings)
        .map_err(|error| format!("Clarabel rejected the 56x56 QP: {error}"))?;
    solver.solve();
    if !matches!(
        solver.solution.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        return Err(format!(
            "Clarabel terminated with status {:?}.",
            solver.solution.status
        ));
    }
    let densities = solver
        .solution
        .x
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    if densities.iter().any(|density| !density.is_finite()) {
        return Err("Clarabel returned a non-finite density solution.".into());
    }
    Ok(densities)
}

fn add_pair_penalty(h: &mut [Vec<f64>], left: usize, right: usize, weight: f64) {
    h[left][left] += weight;
    h[right][right] += weight;
    let (row, col) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    h[row][col] -= weight;
}

fn radial_density_pairs(job: &DensityProblem) -> Vec<(usize, usize)> {
    job.voxels
        .iter()
        .enumerate()
        .flat_map(|(left, a)| {
            job.voxels
                .iter()
                .enumerate()
                .skip(left + 1)
                .filter_map(move |(right, b)| {
                    ((a.center.length() - b.center.length()).abs() <= job.voxel_size * 0.45)
                        .then_some((left, right))
                })
        })
        .collect()
}
