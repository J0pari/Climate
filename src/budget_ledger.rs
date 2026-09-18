//! Streaming extensive-quantity budget accounting.
//!
//! A budget ledger explains one extensive quantity over one declared time
//! interval. Physical kernels remain responsible for computing their own fluxes,
//! sources, sinks, exchanges, corrections, and solver effects; this module owns
//! the common accounting semantics that allow those terms to be accumulated and
//! merged across partitions without materializing a global climate state.
//!
//! All recorded amounts use one sign convention: a positive amount contributes
//! positively to Q_after - Q_before. The ledger never repairs state and never
//! turns an unexplained residual into a named physical process.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetIdentity {
    pub budget_id: String,
    pub quantity_id: String,
    pub units: String,
    pub domain_id: String,
    pub interval_start_ns: i64,
    pub interval_end_ns: i64,
    pub precision: String,
    pub accumulation_method: String,
}

impl BudgetIdentity {
    pub fn new(
        budget_id: impl Into<String>,
        quantity_id: impl Into<String>,
        units: impl Into<String>,
        domain_id: impl Into<String>,
        interval_start_ns: i64,
        interval_end_ns: i64,
        precision: impl Into<String>,
    ) -> Result<Self, BudgetError> {
        let identity = Self {
            budget_id: budget_id.into(),
            quantity_id: quantity_id.into(),
            units: units.into(),
            domain_id: domain_id.into(),
            interval_start_ns,
            interval_end_ns,
            precision: precision.into(),
            accumulation_method: "neumaier_compensated_f64".to_string(),
        };
        identity.validate()?;
        Ok(identity)
    }

    fn validate(&self) -> Result<(), BudgetError> {
        for (field, value) in [
            ("budget_id", self.budget_id.as_str()),
            ("quantity_id", self.quantity_id.as_str()),
            ("units", self.units.as_str()),
            ("domain_id", self.domain_id.as_str()),
            ("precision", self.precision.as_str()),
            ("accumulation_method", self.accumulation_method.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(BudgetError::EmptyIdentityField { field });
            }
        }
        if self.interval_end_ns <= self.interval_start_ns {
            return Err(BudgetError::InvalidInterval {
                start_ns: self.interval_start_ns,
                end_ns: self.interval_end_ns,
            });
        }
        Ok(())
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum BudgetTermClass {
    BoundaryFlux,
    ResolvedPhysicalSource,
    ResolvedPhysicalSink,
    ResolvedDissipation,
    InternalExchange,
    CouplingExchange,
    NumericalCorrection,
    SolverResidualEffect,
    RoundoffEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetContribution {
    pub class: BudgetTermClass,
    pub process_id: String,
    pub boundary_classification: Option<String>,
    /// Signed contribution to Q_after - Q_before in the ledger quantity units.
    pub signed_amount: f64,
    /// Conservative absolute uncertainty/error bound in the same units.
    pub absolute_uncertainty_bound: Option<f64>,
}

impl BudgetContribution {
    pub fn new(
        class: BudgetTermClass,
        process_id: impl Into<String>,
        signed_amount: f64,
    ) -> Self {
        Self {
            class,
            process_id: process_id.into(),
            boundary_classification: None,
            signed_amount,
            absolute_uncertainty_bound: None,
        }
    }

    pub fn with_boundary_classification(
        mut self,
        classification: impl Into<String>,
    ) -> Self {
        self.boundary_classification = Some(classification.into());
        self
    }

    pub fn with_absolute_uncertainty_bound(mut self, bound: f64) -> Self {
        self.absolute_uncertainty_bound = Some(bound);
        self
    }

    fn validate(&self) -> Result<(), BudgetError> {
        if self.process_id.trim().is_empty() {
            return Err(BudgetError::EmptyProcessId);
        }
        if !self.signed_amount.is_finite() {
            return Err(BudgetError::NonFiniteAmount {
                process_id: self.process_id.clone(),
                amount: self.signed_amount,
            });
        }
        if self.class == BudgetTermClass::ResolvedPhysicalSource && self.signed_amount < 0.0 {
            return Err(BudgetError::SourceHasNegativeSign {
                process_id: self.process_id.clone(),
                amount: self.signed_amount,
            });
        }
        if self.class == BudgetTermClass::ResolvedPhysicalSink && self.signed_amount > 0.0 {
            return Err(BudgetError::SinkHasPositiveSign {
                process_id: self.process_id.clone(),
                amount: self.signed_amount,
            });
        }
        if self.class == BudgetTermClass::ResolvedDissipation && self.signed_amount > 0.0 {
            return Err(BudgetError::DissipationHasPositiveSign {
                process_id: self.process_id.clone(),
                amount: self.signed_amount,
            });
        }
        match (&self.class, &self.boundary_classification) {
            (BudgetTermClass::BoundaryFlux, Some(value)) if value.trim().is_empty() => {
                return Err(BudgetError::EmptyBoundaryClassification);
            }
            (BudgetTermClass::BoundaryFlux, None) => {
                return Err(BudgetError::MissingBoundaryClassification);
            }
            (_, Some(value)) if value.trim().is_empty() => {
                return Err(BudgetError::EmptyBoundaryClassification);
            }
            _ => {}
        }
        if let Some(bound) = self.absolute_uncertainty_bound {
            if !bound.is_finite() || bound < 0.0 {
                return Err(BudgetError::InvalidUncertaintyBound {
                    process_id: self.process_id.clone(),
                    bound,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ContributionKey {
    class: BudgetTermClass,
    process_id: String,
    boundary_classification: Option<String>,
}

impl From<&BudgetContribution> for ContributionKey {
    fn from(value: &BudgetContribution) -> Self {
        Self {
            class: value.class,
            process_id: value.process_id.clone(),
            boundary_classification: value.boundary_classification.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn zero() -> Self {
        Self {
            sum: 0.0,
            correction: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Debug, Clone, Copy)]
struct ContributionAccumulator {
    amount: CompensatedSum,
    uncertainty_bound: CompensatedSum,
    has_uncertainty: bool,
}

impl ContributionAccumulator {
    fn zero() -> Self {
        Self {
            amount: CompensatedSum::zero(),
            uncertainty_bound: CompensatedSum::zero(),
            has_uncertainty: false,
        }
    }

    fn add(&mut self, amount: f64, uncertainty: Option<f64>) {
        self.amount.add(amount);
        if let Some(bound) = uncertainty {
            self.uncertainty_bound.add(bound);
            self.has_uncertainty = true;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetTermTotal {
    pub class: BudgetTermClass,
    pub process_id: String,
    pub boundary_classification: Option<String>,
    pub signed_amount: f64,
    pub absolute_uncertainty_bound: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetClassTotal {
    pub class: BudgetTermClass,
    pub signed_amount: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetReport {
    pub identity: BudgetIdentity,
    pub partition_count: usize,
    pub quantity_before: f64,
    pub quantity_after: f64,
    pub actual_change: f64,
    pub accounted_change: f64,
    pub unexplained_residual: f64,
    pub conservative_absolute_uncertainty_bound: Option<f64>,
    pub class_totals: Vec<BudgetClassTotal>,
    pub term_totals: Vec<BudgetTermTotal>,
}

#[derive(Debug, Clone)]
pub struct ExtensiveBudgetLedger {
    identity: BudgetIdentity,
    partition_ids: BTreeSet<String>,
    quantity_before: CompensatedSum,
    quantity_after: CompensatedSum,
    contributions: BTreeMap<ContributionKey, ContributionAccumulator>,
}

impl ExtensiveBudgetLedger {
    pub fn new(
        identity: BudgetIdentity,
        partition_id: impl Into<String>,
        quantity_before: f64,
        quantity_after: f64,
    ) -> Result<Self, BudgetError> {
        identity.validate()?;
        let partition_id = partition_id.into();
        if partition_id.trim().is_empty() {
            return Err(BudgetError::EmptyPartitionId);
        }
        if !quantity_before.is_finite() || !quantity_after.is_finite() {
            return Err(BudgetError::NonFiniteStateTotal {
                before: quantity_before,
                after: quantity_after,
            });
        }
        let mut partition_ids = BTreeSet::new();
        partition_ids.insert(partition_id);
        let mut before = CompensatedSum::zero();
        before.add(quantity_before);
        let mut after = CompensatedSum::zero();
        after.add(quantity_after);
        Ok(Self {
            identity,
            partition_ids,
            quantity_before: before,
            quantity_after: after,
            contributions: BTreeMap::new(),
        })
    }

    pub fn record(&mut self, contribution: BudgetContribution) -> Result<(), BudgetError> {
        contribution.validate()?;
        let key = ContributionKey::from(&contribution);
        self.contributions.entry(key).or_insert_with(ContributionAccumulator::zero).add(
            contribution.signed_amount,
            contribution.absolute_uncertainty_bound,
        );
        Ok(())
    }

    /// Merge a disjoint spatial/data partition covering the same quantity,
    /// domain identity, and time interval.
    ///
    /// Callers remain responsible for assigning partition IDs that describe
    /// non-overlapping extensive state. Duplicate IDs are rejected to prevent
    /// an accidental double count from looking like physical forcing.
    pub fn merge(&mut self, other: &Self) -> Result<(), BudgetError> {
        if self.identity != other.identity {
            return Err(BudgetError::IncompatibleLedgerIdentity);
        }
        if let Some(duplicate) = self
            .partition_ids
            .intersection(&other.partition_ids)
            .next()
            .cloned()
        {
            return Err(BudgetError::DuplicatePartition { partition_id: duplicate });
        }

        self.quantity_before.add(other.quantity_before.total());
        self.quantity_after.add(other.quantity_after.total());
        self.partition_ids.extend(other.partition_ids.iter().cloned());

        for (key, accumulator) in &other.contributions {
            let target = self.contributions.entry(key.clone()).or_insert_with(ContributionAccumulator::zero);
            target.amount.add(accumulator.amount.total());
            if accumulator.has_uncertainty {
                target
                    .uncertainty_bound
                    .add(accumulator.uncertainty_bound.total());
                target.has_uncertainty = true;
            }
        }
        Ok(())
    }

    pub fn report(&self) -> BudgetReport {
        let quantity_before = self.quantity_before.total();
        let quantity_after = self.quantity_after.total();
        let actual_change = quantity_after - quantity_before;

        let mut accounted = CompensatedSum::zero();
        let mut uncertainty = CompensatedSum::zero();
        let mut has_uncertainty = false;
        let mut by_class: BTreeMap<BudgetTermClass, CompensatedSum> = BTreeMap::new();
        let mut term_totals = Vec::with_capacity(self.contributions.len());

        for (key, value) in &self.contributions {
            let amount = value.amount.total();
            accounted.add(amount);
            by_class
                .entry(key.class)
                .or_insert_with(CompensatedSum::zero)
                .add(amount);
            let uncertainty_bound = if value.has_uncertainty {
                let bound = value.uncertainty_bound.total();
                uncertainty.add(bound);
                has_uncertainty = true;
                Some(bound)
            } else {
                None
            };
            term_totals.push(BudgetTermTotal {
                class: key.class,
                process_id: key.process_id.clone(),
                boundary_classification: key.boundary_classification.clone(),
                signed_amount: amount,
                absolute_uncertainty_bound: uncertainty_bound,
            });
        }

        let accounted_change = accounted.total();
        BudgetReport {
            identity: self.identity.clone(),
            partition_count: self.partition_ids.len(),
            quantity_before,
            quantity_after,
            actual_change,
            accounted_change,
            unexplained_residual: actual_change - accounted_change,
            conservative_absolute_uncertainty_bound: has_uncertainty
                .then(|| uncertainty.total()),
            class_totals: by_class
                .into_iter()
                .map(|(class, total)| BudgetClassTotal {
                    class,
                    signed_amount: total.total(),
                })
                .collect(),
            term_totals,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum BudgetError {
    #[error("budget identity field {field} must not be empty")]
    EmptyIdentityField { field: &'static str },
    #[error("budget interval must satisfy end > start; got start={start_ns}, end={end_ns}")]
    InvalidInterval { start_ns: i64, end_ns: i64 },
    #[error("partition_id must not be empty")]
    EmptyPartitionId,
    #[error("process_id must not be empty")]
    EmptyProcessId,
    #[error("budget state totals must be finite; before={before}, after={after}")]
    NonFiniteStateTotal { before: f64, after: f64 },
    #[error("budget contribution from {process_id:?} is non-finite: {amount}")]
    NonFiniteAmount { process_id: String, amount: f64 },
    #[error("physical source {process_id:?} must have nonnegative signed amount; got {amount}")]
    SourceHasNegativeSign { process_id: String, amount: f64 },
    #[error("physical sink {process_id:?} must have nonpositive signed amount; got {amount}")]
    SinkHasPositiveSign { process_id: String, amount: f64 },
    #[error("resolved dissipation {process_id:?} must have nonpositive signed amount; got {amount}")]
    DissipationHasPositiveSign { process_id: String, amount: f64 },
    #[error("boundary flux requires a nonempty boundary classification")]
    MissingBoundaryClassification,
    #[error("boundary classification must not be empty")]
    EmptyBoundaryClassification,
    #[error("uncertainty bound from {process_id:?} must be finite and nonnegative; got {bound}")]
    InvalidUncertaintyBound { process_id: String, bound: f64 },
    #[error("budget ledgers have incompatible quantity/domain/time/precision identity")]
    IncompatibleLedgerIdentity,
    #[error("partition {partition_id:?} would be counted more than once")]
    DuplicatePartition { partition_id: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> BudgetIdentity {
        BudgetIdentity::new(
            "dry_energy.step.0001",
            "dry_total_energy",
            "J",
            "global_atmosphere",
            0,
            300_000_000_000,
            "fp64",
        )
        .unwrap()
    }

    fn approx(left: f64, right: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= 2.0e-14 * scale,
            "{left:e} != {right:e}"
        );
    }

    #[test]
    fn closed_internal_exchange_cancels_without_becoming_a_source() {
        let mut ledger =
            ExtensiveBudgetLedger::new(identity(), "column-0", 100.0, 100.0).unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::InternalExchange,
                "pressure_work.kinetic_to_internal",
                -7.5,
            ))
            .unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::InternalExchange,
                "pressure_work.internal_from_kinetic",
                7.5,
            ))
            .unwrap();

        let report = ledger.report();
        approx(report.actual_change, 0.0);
        approx(report.accounted_change, 0.0);
        approx(report.unexplained_residual, 0.0);
        let internal = report
            .class_totals
            .iter()
            .find(|item| item.class == BudgetTermClass::InternalExchange)
            .unwrap();
        approx(internal.signed_amount, 0.0);
    }

    #[test]
    fn forcing_sink_and_boundary_flux_explain_state_change() {
        let mut ledger =
            ExtensiveBudgetLedger::new(identity(), "column-0", 100.0, 111.0).unwrap();
        ledger
            .record(
                BudgetContribution::new(
                    BudgetTermClass::BoundaryFlux,
                    "surface.sensible_heat",
                    10.0,
                )
                .with_boundary_classification("lower_boundary"),
            )
            .unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::ResolvedPhysicalSource,
                "radiation.shortwave_absorption",
                2.0,
            ))
            .unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::ResolvedPhysicalSink,
                "radiation.longwave_export",
                -1.0,
            ))
            .unwrap();

        let report = ledger.report();
        approx(report.actual_change, 11.0);
        approx(report.accounted_change, 11.0);
        approx(report.unexplained_residual, 0.0);
    }

    #[test]
    fn numerical_correction_remains_visible() {
        let mut ledger =
            ExtensiveBudgetLedger::new(identity(), "column-0", 10.0, 9.75).unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::NumericalCorrection,
                "positivity_repair",
                -0.25,
            ))
            .unwrap();
        let report = ledger.report();
        approx(report.unexplained_residual, 0.0);
        assert_eq!(
            report.term_totals[0].class,
            BudgetTermClass::NumericalCorrection
        );
    }

    #[test]
    fn coupled_channels_keep_exchange_forcing_dissipation_and_correction_distinct() {
        let mut ledger =
            ExtensiveBudgetLedger::new(identity(), "column-0", 100.0, 106.75).unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::InternalExchange,
                "pressure_work.kinetic",
                -5.0,
            ))
            .unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::InternalExchange,
                "pressure_work.thermodynamic",
                5.0,
            ))
            .unwrap();
        ledger
            .record(
                BudgetContribution::new(
                    BudgetTermClass::BoundaryFlux,
                    "surface_forcing",
                    10.0,
                )
                .with_boundary_classification("lower_boundary"),
            )
            .unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::ResolvedDissipation,
                "vertical_diffusion.quadratic",
                -3.0,
            ))
            .unwrap();
        ledger
            .record(BudgetContribution::new(
                BudgetTermClass::NumericalCorrection,
                "positivity_repair",
                -0.25,
            ))
            .unwrap();

        let report = ledger.report();
        approx(report.actual_change, 6.75);
        approx(report.accounted_change, 6.75);
        approx(report.unexplained_residual, 0.0);
        for (class, expected) in [
            (BudgetTermClass::InternalExchange, 0.0),
            (BudgetTermClass::BoundaryFlux, 10.0),
            (BudgetTermClass::ResolvedDissipation, -3.0),
            (BudgetTermClass::NumericalCorrection, -0.25),
        ] {
            let total = report
                .class_totals
                .iter()
                .find(|item| item.class == class)
                .unwrap();
            approx(total.signed_amount, expected);
        }
    }

    #[test]
    fn unexplained_residual_is_reported_not_repaired() {
        let ledger =
            ExtensiveBudgetLedger::new(identity(), "column-0", 10.0, 10.125).unwrap();
        let report = ledger.report();
        approx(report.actual_change, 0.125);
        approx(report.accounted_change, 0.0);
        approx(report.unexplained_residual, 0.125);
    }

    #[test]
    fn partition_merge_matches_monolithic_accounting() {
        let mut left =
            ExtensiveBudgetLedger::new(identity(), "west", 40.0, 46.0).unwrap();
        left.record(BudgetContribution::new(
            BudgetTermClass::ResolvedPhysicalSource,
            "radiation",
            6.0,
        ))
        .unwrap();

        let mut right =
            ExtensiveBudgetLedger::new(identity(), "east", 60.0, 63.0).unwrap();
        right
            .record(
                BudgetContribution::new(
                    BudgetTermClass::BoundaryFlux,
                    "surface_exchange",
                    4.0,
                )
                .with_boundary_classification("lower_boundary"),
            )
            .unwrap();
        right
            .record(BudgetContribution::new(
                BudgetTermClass::ResolvedPhysicalSink,
                "top_export",
                -1.0,
            ))
            .unwrap();

        left.merge(&right).unwrap();
        let merged = left.report();

        let mut monolithic =
            ExtensiveBudgetLedger::new(identity(), "all", 100.0, 109.0).unwrap();
        monolithic
            .record(BudgetContribution::new(
                BudgetTermClass::ResolvedPhysicalSource,
                "radiation",
                6.0,
            ))
            .unwrap();
        monolithic
            .record(
                BudgetContribution::new(
                    BudgetTermClass::BoundaryFlux,
                    "surface_exchange",
                    4.0,
                )
                .with_boundary_classification("lower_boundary"),
            )
            .unwrap();
        monolithic
            .record(BudgetContribution::new(
                BudgetTermClass::ResolvedPhysicalSink,
                "top_export",
                -1.0,
            ))
            .unwrap();
        let single = monolithic.report();

        approx(merged.quantity_before, single.quantity_before);
        approx(merged.quantity_after, single.quantity_after);
        approx(merged.accounted_change, single.accounted_change);
        approx(merged.unexplained_residual, single.unexplained_residual);
        assert_eq!(merged.partition_count, 2);
    }

    #[test]
    fn duplicate_partition_is_rejected_before_double_counting() {
        let mut left =
            ExtensiveBudgetLedger::new(identity(), "same", 1.0, 1.0).unwrap();
        let right =
            ExtensiveBudgetLedger::new(identity(), "same", 1.0, 1.0).unwrap();
        assert!(matches!(
            left.merge(&right),
            Err(BudgetError::DuplicatePartition { .. })
        ));
    }

    #[test]
    fn malformed_accounting_fails_closed() {
        let mut ledger =
            ExtensiveBudgetLedger::new(identity(), "column-0", 1.0, 1.0).unwrap();

        assert!(matches!(
            ledger.record(BudgetContribution::new(
                BudgetTermClass::ResolvedPhysicalSource,
                "source",
                -1.0,
            )),
            Err(BudgetError::SourceHasNegativeSign { .. })
        ));
        assert!(matches!(
            ledger.record(BudgetContribution::new(
                BudgetTermClass::ResolvedPhysicalSink,
                "sink",
                1.0,
            )),
            Err(BudgetError::SinkHasPositiveSign { .. })
        ));
        assert!(matches!(
            ledger.record(BudgetContribution::new(
                BudgetTermClass::BoundaryFlux,
                "boundary",
                1.0,
            )),
            Err(BudgetError::MissingBoundaryClassification)
        ));
        assert!(matches!(
            ledger.record(BudgetContribution::new(
                BudgetTermClass::ResolvedDissipation,
                "diffusion",
                1.0,
            )),
            Err(BudgetError::DissipationHasPositiveSign { .. })
        ));
    }
}
