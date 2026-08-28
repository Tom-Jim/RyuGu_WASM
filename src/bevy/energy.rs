use crate::cpu::curved_arc::CurvedArcResidualHistory;
use crate::interface::components::*;
use crate::interface::select_history;
use bevy::prelude::*;

include!("energy/jacobi_math.rs");
include!("energy/jacobi_backend.rs");
