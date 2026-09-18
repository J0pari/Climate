//! Climate-specific adapters from physical-kernel diagnostics to the common ledger.
//!
//! Physical kernels own the equations and the signed scalar terms they expose.
//! The generic ledger owns accumulation and reporting.  This module is the narrow
//! semantic seam between them; it does not call a solver or infer terms from
//! state mutations.

use crate::budget_ledger::{
    BudgetContribution, BudgetError, BudgetIdentity, BudgetTermClass,
    ExtensiveBudgetLedger,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportQuantity {
    CarrierMass,
    TracerMass,
}

impl TransportQuantity {
    fn process_prefix(self) -> &'static str {
        match self {
            Self::CarrierMass => "transport.carrier_mass",
            Self::TracerMass => "transport.tracer_mass",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservativeTransportBudgetTerms {
    pub quantity_before_kg: f64,
    pub quantity_after_kg: f64,
    pub lower_boundary_change_kg: f64,
    pub upper_boundary_change_kg: f64,
    pub resolved_source_change_kg: f64,
    pub resolved_sink_change_kg: f64,
}

pub fn conservative_transport_ledger(
    identity: BudgetIdentity,
    partition_id: impl Into<String>,
    quantity: TransportQuantity,
    terms: ConservativeTransportBudgetTerms,
) -> Result<ExtensiveBudgetLedger, PhysicalBudgetAdapterError> {
    require_units(&identity, "kg")?;
    if !terms.resolved_source_change_kg.is_finite()
        || terms.resolved_source_change_kg < 0.0
    {
        return Err(PhysicalBudgetAdapterError::InvalidResolvedSource {
            amount: terms.resolved_source_change_kg,
        });
    }
    if !terms.resolved_sink_change_kg.is_finite()
        || terms.resolved_sink_change_kg > 0.0
    {
        return Err(PhysicalBudgetAdapterError::InvalidResolvedSink {
            amount: terms.resolved_sink_change_kg,
        });
    }

    let prefix = quantity.process_prefix();
    let mut ledger = ExtensiveBudgetLedger::new(
        identity,
        partition_id,
        terms.quantity_before_kg,
        terms.quantity_after_kg,
    )?;
    ledger.record(
        BudgetContribution::new(
            BudgetTermClass::BoundaryFlux,
            format!("{prefix}.lower_boundary"),
            terms.lower_boundary_change_kg,
        )
        .with_boundary_classification("lower_boundary"),
    )?;
    ledger.record(
        BudgetContribution::new(
            BudgetTermClass::BoundaryFlux,
            format!("{prefix}.upper_boundary"),
            terms.upper_boundary_change_kg,
        )
        .with_boundary_classification("upper_boundary"),
    )?;
    ledger.record(BudgetContribution::new(
        BudgetTermClass::ResolvedPhysicalSource,
        format!("{prefix}.resolved_source"),
        terms.resolved_source_change_kg,
    ))?;
    ledger.record(BudgetContribution::new(
        BudgetTermClass::ResolvedPhysicalSink,
        format!("{prefix}.resolved_sink"),
        terms.resolved_sink_change_kg,
    ))?;
    Ok(ledger)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerticalDiffusionInventoryBudgetTerms {
    pub quantity_before: f64,
    pub quantity_after: f64,
    pub lower_boundary_change: f64,
    pub upper_boundary_change: f64,
}

pub fn vertical_diffusion_inventory_ledger(
    identity: BudgetIdentity,
    partition_id: impl Into<String>,
    terms: VerticalDiffusionInventoryBudgetTerms,
) -> Result<ExtensiveBudgetLedger, PhysicalBudgetAdapterError> {
    let mut ledger = ExtensiveBudgetLedger::new(
        identity,
        partition_id,
        terms.quantity_before,
        terms.quantity_after,
    )?;
    ledger.record(
        BudgetContribution::new(
            BudgetTermClass::BoundaryFlux,
            "vertical_diffusion.inventory.lower_boundary",
            terms.lower_boundary_change,
        )
        .with_boundary_classification("lower_boundary"),
    )?;
    ledger.record(
        BudgetContribution::new(
            BudgetTermClass::BoundaryFlux,
            "vertical_diffusion.inventory.upper_boundary",
            terms.upper_boundary_change,
        )
        .with_boundary_classification("upper_boundary"),
    )?;
    Ok(ledger)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerticalDiffusionQuadraticBudgetTerms {
    pub quantity_before: f64,
    pub quantity_after: f64,
    pub lower_boundary_exchange: f64,
    pub upper_boundary_exchange: f64,
    pub interior_dissipation: f64,
}

pub fn vertical_diffusion_quadratic_ledger(
    identity: BudgetIdentity,
    partition_id: impl Into<String>,
    terms: VerticalDiffusionQuadraticBudgetTerms,
) -> Result<ExtensiveBudgetLedger, PhysicalBudgetAdapterError> {
    let mut ledger = ExtensiveBudgetLedger::new(
        identity,
        partition_id,
        terms.quantity_before,
        terms.quantity_after,
    )?;
    ledger.record(
        BudgetContribution::new(
            BudgetTermClass::BoundaryFlux,
            "vertical_diffusion.quadratic.lower_boundary",
            terms.lower_boundary_exchange,
        )
        .with_boundary_classification("lower_boundary"),
    )?;
    ledger.record(
        BudgetContribution::new(
            BudgetTermClass::BoundaryFlux,
            "vertical_diffusion.quadratic.upper_boundary",
            terms.upper_boundary_exchange,
        )
        .with_boundary_classification("upper_boundary"),
    )?;
    ledger.record(BudgetContribution::new(
        BudgetTermClass::ResolvedDissipation,
        "vertical_diffusion.quadratic.interior",
        terms.interior_dissipation,
    ))?;
    Ok(ledger)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureEnergyExchangeTerms {
    pub kinetic_pressure_gradient_power_w_kg: f64,
    pub enthalpy_pressure_work_w_kg: f64,
    pub geopotential_material_tendency_w_kg: f64,
    pub local_geopotential_tendency_w_kg: f64,
}

pub fn pressure_energy_exchange_ledger(
    identity: BudgetIdentity,
    partition_id: impl Into<String>,
    quantity_before_j: f64,
    quantity_after_j: f64,
    mass_kg: f64,
    dt_s: f64,
    terms: PressureEnergyExchangeTerms,
) -> Result<ExtensiveBudgetLedger, PhysicalBudgetAdapterError> {
    require_units(&identity, "J")?;
    if !mass_kg.is_finite() || mass_kg <= 0.0 {
        return Err(PhysicalBudgetAdapterError::InvalidMass { mass_kg });
    }
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(PhysicalBudgetAdapterError::InvalidTimestep { dt_s });
    }

    let scale = mass_kg * dt_s;
    if !scale.is_finite() {
        return Err(PhysicalBudgetAdapterError::NonFiniteScale);
    }

    // The material geopotential tendency contains both the exchange term and
    // any explicit local geopotential tendency.  Subtract the latter before
    // labeling the remainder as internal, so the three internal reservoir
    // transfers cancel when the Fortran closure relation is satisfied.
    let geopotential_internal =
        terms.geopotential_material_tendency_w_kg
            - terms.local_geopotential_tendency_w_kg;

    let mut ledger = ExtensiveBudgetLedger::new(
        identity,
        partition_id,
        quantity_before_j,
        quantity_after_j,
    )?;
    for (process_id, specific_power) in [
        (
            "pressure_exchange.kinetic_pressure_gradient",
            terms.kinetic_pressure_gradient_power_w_kg,
        ),
        (
            "pressure_exchange.enthalpy_pressure_work",
            terms.enthalpy_pressure_work_w_kg,
        ),
        (
            "pressure_exchange.geopotential_internal",
            geopotential_internal,
        ),
    ] {
        ledger.record(BudgetContribution::new(
            BudgetTermClass::InternalExchange,
            process_id,
            specific_power * scale,
        ))?;
    }
    ledger.record(BudgetContribution::new(
        BudgetTermClass::CouplingExchange,
        "pressure_exchange.local_geopotential_tendency",
        terms.local_geopotential_tendency_w_kg * scale,
    ))?;
    Ok(ledger)
}

fn require_units(
    identity: &BudgetIdentity,
    expected: &'static str,
) -> Result<(), PhysicalBudgetAdapterError> {
    if identity.units != expected {
        return Err(PhysicalBudgetAdapterError::UnexpectedUnits {
            expected,
            actual: identity.units.clone(),
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum PhysicalBudgetAdapterError {
    #[error(transparent)]
    Budget(#[from] BudgetError),
    #[error("physical budget adapter expected units {expected:?}, got {actual:?}")]
    UnexpectedUnits {
        expected: &'static str,
        actual: String,
    },
    #[error("resolved transport source must be finite and nonnegative; got {amount}")]
    InvalidResolvedSource { amount: f64 },
    #[error("resolved transport sink must be finite and nonpositive; got {amount}")]
    InvalidResolvedSink { amount: f64 },
    #[error("pressure-energy adapter mass must be finite and positive; got {mass_kg}")]
    InvalidMass { mass_kg: f64 },
    #[error("pressure-energy adapter timestep must be finite and positive; got {dt_s}")]
    InvalidTimestep { dt_s: f64 },
    #[error("pressure-energy mass-times-timestep scale is non-finite")]
    NonFiniteScale,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget_ledger::BudgetTermClass;

    fn identity(quantity: &str, units: &str) -> BudgetIdentity {
        BudgetIdentity::new(
            format!("{quantity}.step.0001"),
            quantity,
            units,
            "column-0",
            0,
            2_000_000_000,
            "fp64",
        )
        .unwrap()
    }

    fn approx(left: f64, right: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!((left - right).abs() <= 2.0e-14 * scale, "{left} != {right}");
    }

    #[test]
    fn conservative_transport_terms_map_without_hiding_residual() {
        let ledger = conservative_transport_ledger(
            identity("carrier_mass", "kg"),
            "column-0",
            TransportQuantity::CarrierMass,
            ConservativeTransportBudgetTerms {
                quantity_before_kg: 20.0,
                quantity_after_kg: 21.3,
                lower_boundary_change_kg: 2.0,
                upper_boundary_change_kg: -0.5,
                resolved_source_change_kg: 0.2,
                resolved_sink_change_kg: -0.4,
            },
        )
        .unwrap();
        let report = ledger.report();
        approx(report.actual_change, 1.3);
        approx(report.accounted_change, 1.3);
        approx(report.unexplained_residual, 0.0);
        assert_eq!(
            report
                .class_totals
                .iter()
                .find(|item| item.class == BudgetTermClass::BoundaryFlux)
                .unwrap()
                .signed_amount,
            1.5
        );
    }

    #[test]
    fn tracer_terms_use_the_same_ledger_semantics() {
        let ledger = conservative_transport_ledger(
            identity("tracer_mass", "kg"),
            "column-0",
            TransportQuantity::TracerMass,
            ConservativeTransportBudgetTerms {
                quantity_before_kg: 3.0,
                quantity_after_kg: 3.02,
                lower_boundary_change_kg: 0.4,
                upper_boundary_change_kg: -0.4,
                resolved_source_change_kg: 0.04,
                resolved_sink_change_kg: -0.02,
            },
        )
        .unwrap();
        let report = ledger.report();
        approx(report.accounted_change, 0.02);
        approx(report.unexplained_residual, 0.0);
    }

    #[test]
    fn vertical_diffusion_inventory_records_only_boundary_exchange() {
        let ledger = vertical_diffusion_inventory_ledger(
            identity("column_scalar_inventory", "scalar m"),
            "column-0",
            VerticalDiffusionInventoryBudgetTerms {
                quantity_before: 10.0,
                quantity_after: 11.5,
                lower_boundary_change: 2.0,
                upper_boundary_change: -0.5,
            },
        )
        .unwrap();
        let report = ledger.report();
        approx(report.actual_change, 1.5);
        approx(report.accounted_change, 1.5);
        approx(report.unexplained_residual, 0.0);
        assert_eq!(report.class_totals.len(), 1);
        assert_eq!(report.class_totals[0].class, BudgetTermClass::BoundaryFlux);
    }

    #[test]
    fn vertical_diffusion_quadratic_keeps_dissipation_separate_from_boundaries() {
        let ledger = vertical_diffusion_quadratic_ledger(
            identity("weighted_scalar_quadratic", "scalar^2 m"),
            "column-0",
            VerticalDiffusionQuadraticBudgetTerms {
                quantity_before: 10.0,
                quantity_after: 8.5,
                lower_boundary_exchange: 1.0,
                upper_boundary_exchange: -0.5,
                interior_dissipation: -2.0,
            },
        )
        .unwrap();
        let report = ledger.report();
        approx(report.actual_change, -1.5);
        approx(report.accounted_change, -1.5);
        approx(report.unexplained_residual, 0.0);
        let dissipation = report
            .class_totals
            .iter()
            .find(|item| item.class == BudgetTermClass::ResolvedDissipation)
            .unwrap();
        approx(dissipation.signed_amount, -2.0);
    }

    #[test]
    fn vertical_diffusion_positive_dissipation_fails_closed() {
        let error = vertical_diffusion_quadratic_ledger(
            identity("weighted_scalar_quadratic", "scalar^2 m"),
            "column-0",
            VerticalDiffusionQuadraticBudgetTerms {
                quantity_before: 1.0,
                quantity_after: 1.0,
                lower_boundary_exchange: 0.0,
                upper_boundary_exchange: 0.0,
                interior_dissipation: 0.1,
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PhysicalBudgetAdapterError::Budget(
                BudgetError::DissipationHasPositiveSign { .. }
            )
        ));
    }

    #[test]
    fn pressure_exchange_separates_internal_cancellation_from_local_coupling() {
        // Same scalar diagnostic values as the canonical Fortran
        // local-geopotential-tendency witness.
        let terms = PressureEnergyExchangeTerms {
            kinetic_pressure_gradient_power_w_kg: 0.0044,
            enthalpy_pressure_work_w_kg: 0.1125,
            geopotential_material_tendency_w_kg: -0.1144,
            local_geopotential_tendency_w_kg: 0.0025,
        };
        let ledger = pressure_energy_exchange_ledger(
            identity("dry_total_energy", "J"),
            "column-0",
            100.0,
            100.02,
            2.0,
            4.0,
            terms,
        )
        .unwrap();
        let report = ledger.report();
        approx(report.actual_change, 0.02);
        approx(report.accounted_change, 0.02);
        approx(report.unexplained_residual, 0.0);

        let internal = report
            .class_totals
            .iter()
            .find(|item| item.class == BudgetTermClass::InternalExchange)
            .unwrap();
        approx(internal.signed_amount, 0.0);
        let coupling = report
            .class_totals
            .iter()
            .find(|item| item.class == BudgetTermClass::CouplingExchange)
            .unwrap();
        approx(coupling.signed_amount, 0.02);
    }

    #[test]
    fn inconsistent_pressure_exchange_remains_an_unexplained_residual() {
        let ledger = pressure_energy_exchange_ledger(
            identity("dry_total_energy", "J"),
            "column-0",
            10.0,
            10.0,
            1.0,
            1.0,
            PressureEnergyExchangeTerms {
                kinetic_pressure_gradient_power_w_kg: 1.0,
                enthalpy_pressure_work_w_kg: 2.0,
                geopotential_material_tendency_w_kg: -2.5,
                local_geopotential_tendency_w_kg: 0.0,
            },
        )
        .unwrap();
        approx(ledger.report().unexplained_residual, -0.5);
    }
}
