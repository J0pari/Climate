//! Canonical compile-checked Rust surface for Climate.
//!
//! Canonical implementations live under `src/`; historical prototype source is
//! available through Git history rather than a compatibility feature surface.

pub mod clifford;
pub mod ebm_observation_geometry;
pub mod feedback;
pub mod geometry;
pub mod inference;
pub mod modal;
pub mod numerics;
pub mod padic;
pub mod physical_state;
pub mod sheaf;
