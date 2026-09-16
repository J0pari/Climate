//! Canonical compile-checked Rust surface for Climate.
//!
//! Canonical implementations live under `src/`. Root-level Rust prototypes are
//! retained as migration/comparison material and can be compile-checked through
//! the `legacy-rust` feature without making their historical tests or semantics
//! authoritative for the canonical crate.

pub mod clifford;
pub mod feedback;
pub mod geometry;
pub mod modal;
pub mod numerics;
pub mod padic;
pub mod sheaf;

#[cfg(feature = "legacy-rust")]
#[path = "../climate_scenario_logic.rs"]
pub mod climate_scenario_logic;

#[cfg(feature = "legacy-rust")]
#[path = "../climate_feedback_validators.rs"]
pub mod climate_feedback_validators;
