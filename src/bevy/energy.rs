use crate::interface::components::*;
use crate::interface::select_history;
use bevy::platform::time::Instant;
use bevy::prelude::*;
use std::time::Duration;

include!("energy/jacobi_math.rs");
include!("energy/jacobi_backend.rs");
