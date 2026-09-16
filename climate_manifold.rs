//! Climate-state manifold capability stub.
//!
//! The previous implementation was removed from the live execution surface under
//! `docs/SEMANTIC-SANITATION.md`. It combined incomplete dual-number geometry
//! machinery with heuristic mappings from curvature/eigenvalues to named climate
//! tipping probabilities, timescales, and reversibility.
//!
//! Reintroduction requirements:
//!   - `blueprints/geometric-state-manifold.md`
//!   - `docs/GEOMETRY-VERIFICATION.md`
//!   - `experiments/geometry-correctness.v1.json`
//!
//! This module intentionally exports no geometric result type that could be
//! mistaken for a verified climate capability.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClimateManifoldUnavailable;

impl ClimateManifoldUnavailable {
    pub const REASON: &'static str =
        "legacy manifold implementation removed: generic geometry and climate interpretation must be reintroduced through separate verified layers";

    pub fn fail() -> ! {
        panic!("{}; see blueprints/geometric-state-manifold.md", Self::REASON)
    }
}
