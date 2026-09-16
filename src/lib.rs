//! Canonical compile-checked Rust surface for Climate.
//!
//! Legacy root-level prototypes remain available for migration and comparison,
//! but canonical implementations live under `src/` and join this crate only when
//! their dependencies, failure semantics, and tests are explicit.

pub mod numerics;

#[path = "../climate_scenario_logic.rs"]
pub mod climate_scenario_logic;

#[path = "../climate_feedback_validators.rs"]
pub mod climate_feedback_validators;
