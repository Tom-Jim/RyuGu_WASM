include!("ui_controls/control_types.rs");
include!("ui_controls/planning_controls.rs");
include!("ui_controls/planning_batch.rs");
include!("ui_controls/trajectory_controls.rs");
include!("ui_controls/inversion_status.rs");
// planning_controls.rs owns the planning workload selectors and batch-status
// bridge; it is included before status rendering so the shared types resolve.
include!("ui_controls/performance_controls.rs");
include!("ui_controls/inversion_controls.rs");
include!("ui_controls/runtime_panels.rs");
