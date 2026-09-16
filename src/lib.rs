//! Compile-checked Rust surface for Climate.
//!
//! This crate intentionally starts with modules that can be built without
//! CUDA, MPI, Python/Julia FFI, or native data libraries. Additional legacy
//! modules should join only when their dependencies and tests are explicit.

#[path = "../climate_scenario_logic.rs"]
pub mod climate_scenario_logic;

#[path = "../climate_feedback_validators.rs"]
pub mod climate_feedback_validators;
