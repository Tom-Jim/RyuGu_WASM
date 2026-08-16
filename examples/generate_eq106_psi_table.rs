//! Regenerates the certified Eq.106 complex operator embedded by the WASM app.

use ryugu_wasm::{PsiOperatorTable, ToroidalOperatorTensor};

// Maximum angular-cell plane radius after the 900 m model normalization.
const RYUGU_MODEL_RADIUS: f64 = 464.765_191_415_103_6;

fn main() {
    let table = PsiOperatorTable::build(RYUGU_MODEL_RADIUS).expect("build certified Psi table");
    assert!(
        table.validate(3.0e-3),
        "table certificate failed: map={:.3e}, asymptotic={:.3e}, axis={:.3e}",
        table.max_validation_error,
        table.max_asymptotic_remainder,
        table.max_axis_limit_error,
    );
    let mut bytes = Vec::with_capacity(40 + table.coefficients.len() * 4);
    bytes.extend_from_slice(b"EQ106PSI");
    for value in [
        table.radius,
        table.max_validation_error,
        table.max_asymptotic_remainder,
        table.max_axis_limit_error,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &table.coefficients {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::create_dir_all("assets/operators").expect("create operator asset directory");
    std::fs::write("assets/operators/eq106_psi_table.bin", bytes)
        .expect("write certified operator table");
    let toroidal = ToroidalOperatorTensor::build().expect("build toroidal table");
    assert!(toroidal.validate(2.0e-4));
    let mut toroidal_bytes = Vec::with_capacity(16 + toroidal.coefficients.len() * 4);
    toroidal_bytes.extend_from_slice(b"EQ106TOR");
    toroidal_bytes.extend_from_slice(&toroidal.max_midpoint_error.to_le_bytes());
    for value in &toroidal.coefficients {
        toroidal_bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write("assets/operators/eq106_toroidal_table.bin", toroidal_bytes)
        .expect("write certified toroidal operator table");
    println!(
        "wrote {} coefficients; map={:.3e}, asymptotic={:.3e}, axis={:.3e}",
        table.coefficients.len(),
        table.max_validation_error,
        table.max_asymptotic_remainder,
        table.max_axis_limit_error,
    );
    println!(
        "wrote {} toroidal coefficients; midpoint={:.3e}",
        toroidal.coefficients.len(),
        toroidal.max_midpoint_error,
    );
}
