//! Typed boundaries between observations, prognostic fields, and experimental representations.
//!
//! This module owns state-role, availability, initialization provenance, and
//! adapter compatibility identity only. Domain kernels retain ownership of
//! numerical validity, discretization, units interpretation, and field-specific
//! invariants. Values remain generic so a scalar, array, or domain-owned field
//! type can cross the same semantic boundary without introducing a repository-
//! wide state container.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldDescriptor {
    pub field_id: String,
    pub units: String,
    pub coordinates: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AvailabilityReason(String);

impl AvailabilityReason {
    pub fn new(reason: impl Into<String>) -> Result<Self, StateBoundaryError> {
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(StateBoundaryError::EmptyAvailabilityReason);
        }
        Ok(Self(reason))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for AvailabilityReason {
    type Error = StateBoundaryError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<AvailabilityReason> for String {
    fn from(value: AvailabilityReason) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Availability<T> {
    Present { value: T },
    Missing { reason: AvailabilityReason },
    Unavailable { reason: AvailabilityReason },
}

impl<T> Availability<T> {
    pub fn present(value: T) -> Self {
        Self::Present { value }
    }

    pub fn missing(reason: impl Into<String>) -> Result<Self, StateBoundaryError> {
        Ok(Self::Missing {
            reason: AvailabilityReason::new(reason)?,
        })
    }

    pub fn unavailable(reason: impl Into<String>) -> Result<Self, StateBoundaryError> {
        Ok(Self::Unavailable {
            reason: AvailabilityReason::new(reason)?,
        })
    }

    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Present { value } => Some(value),
            Self::Missing { .. } | Self::Unavailable { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationProvenance {
    pub source_id: String,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservedField<T> {
    pub descriptor: FieldDescriptor,
    pub availability: Availability<T>,
    pub provenance: ObservationProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssimilationAdapterContract {
    adapter_id: String,
    semantic_version: String,
    input_field_id: String,
    output_field_id: String,
}

impl AssimilationAdapterContract {
    pub fn new(
        adapter_id: impl Into<String>,
        semantic_version: impl Into<String>,
        input_field_id: impl Into<String>,
        output_field_id: impl Into<String>,
    ) -> Result<Self, StateBoundaryError> {
        let adapter_id = adapter_id.into();
        let semantic_version = semantic_version.into();
        let input_field_id = input_field_id.into();
        let output_field_id = output_field_id.into();
        if adapter_id.trim().is_empty() {
            return Err(StateBoundaryError::EmptyAdapterId);
        }
        if semantic_version.trim().is_empty() {
            return Err(StateBoundaryError::EmptySemanticVersion);
        }
        if input_field_id.trim().is_empty() || output_field_id.trim().is_empty() {
            return Err(StateBoundaryError::EmptyFieldId);
        }
        Ok(Self {
            adapter_id,
            semantic_version,
            input_field_id,
            output_field_id,
        })
    }

    pub fn adapter_id(&self) -> &str {
        &self.adapter_id
    }

    pub fn semantic_version(&self) -> &str {
        &self.semantic_version
    }

    pub fn input_field_id(&self) -> &str {
        &self.input_field_id
    }

    pub fn output_field_id(&self) -> &str {
        &self.output_field_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InitializationProvenance {
    ExplicitModelInput,
    Restart {
        artifact_digest: String,
    },
    AssimilatedObservation {
        adapter_id: String,
        adapter_version: String,
        source_id: String,
        source_artifact_digest: String,
        source_field_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrognosticField<T> {
    pub descriptor: FieldDescriptor,
    pub value: T,
    pub initialization: InitializationProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentalRepresentation<T> {
    pub representation_id: String,
    pub source_field_ids: Vec<String>,
    pub value: T,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StateBoundaryError {
    #[error("field_id must not be empty")]
    EmptyFieldId,
    #[error("units must not be empty")]
    EmptyUnits,
    #[error("coordinates must not be empty")]
    EmptyCoordinates,
    #[error("source_id must not be empty")]
    EmptySourceId,
    #[error("adapter_id must not be empty")]
    EmptyAdapterId,
    #[error("semantic_version must not be empty")]
    EmptySemanticVersion,
    #[error("missing/unavailable reason must not be empty")]
    EmptyAvailabilityReason,
    #[error("experimental representation requires at least one source field")]
    EmptyRepresentationSources,
    #[error("experimental representation repeats source field {field_id:?}")]
    DuplicateSourceFieldId { field_id: String },
    #[error("assimilation adapter expects input field {expected:?}, got {actual:?}")]
    AssimilationInputFieldMismatch { expected: String, actual: String },
    #[error("assimilation adapter produces output field {expected:?}, got {actual:?}")]
    AssimilationOutputFieldMismatch { expected: String, actual: String },
    #[error("artifact digest must use sha256:<64 lowercase hex digits>")]
    InvalidArtifactDigest,
    #[error("assimilation requires a present observation")]
    ObservationNotPresent,
}

fn validate_descriptor(descriptor: &FieldDescriptor) -> Result<(), StateBoundaryError> {
    if descriptor.field_id.trim().is_empty() {
        return Err(StateBoundaryError::EmptyFieldId);
    }
    if descriptor.units.trim().is_empty() {
        return Err(StateBoundaryError::EmptyUnits);
    }
    if descriptor.coordinates.trim().is_empty() {
        return Err(StateBoundaryError::EmptyCoordinates);
    }
    Ok(())
}

fn valid_sha256_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl ObservationProvenance {
    pub fn new(
        source_id: impl Into<String>,
        artifact_digest: impl Into<String>,
    ) -> Result<Self, StateBoundaryError> {
        let source_id = source_id.into();
        let artifact_digest = artifact_digest.into();
        if source_id.trim().is_empty() {
            return Err(StateBoundaryError::EmptySourceId);
        }
        if !valid_sha256_digest(&artifact_digest) {
            return Err(StateBoundaryError::InvalidArtifactDigest);
        }
        Ok(Self {
            source_id,
            artifact_digest,
        })
    }
}

impl<T> ObservedField<T> {
    pub fn new(
        descriptor: FieldDescriptor,
        availability: Availability<T>,
        provenance: ObservationProvenance,
    ) -> Result<Self, StateBoundaryError> {
        validate_descriptor(&descriptor)?;
        Ok(Self {
            descriptor,
            availability,
            provenance,
        })
    }
}

impl<T> PrognosticField<T> {
    pub fn explicit(
        descriptor: FieldDescriptor,
        value: T,
    ) -> Result<Self, StateBoundaryError> {
        validate_descriptor(&descriptor)?;
        Ok(Self {
            descriptor,
            value,
            initialization: InitializationProvenance::ExplicitModelInput,
        })
    }

    pub fn from_restart(
        descriptor: FieldDescriptor,
        value: T,
        artifact_digest: impl Into<String>,
    ) -> Result<Self, StateBoundaryError> {
        validate_descriptor(&descriptor)?;
        let artifact_digest = artifact_digest.into();
        if !valid_sha256_digest(&artifact_digest) {
            return Err(StateBoundaryError::InvalidArtifactDigest);
        }
        Ok(Self {
            descriptor,
            value,
            initialization: InitializationProvenance::Restart { artifact_digest },
        })
    }
}

pub fn assimilated_prognostic_field<O, T>(
    descriptor: FieldDescriptor,
    value: T,
    observation: &ObservedField<O>,
    adapter: &AssimilationAdapterContract,
) -> Result<PrognosticField<T>, StateBoundaryError> {
    validate_descriptor(&descriptor)?;
    if observation.availability.value().is_none() {
        return Err(StateBoundaryError::ObservationNotPresent);
    }
    if observation.descriptor.field_id != adapter.input_field_id {
        return Err(StateBoundaryError::AssimilationInputFieldMismatch {
            expected: adapter.input_field_id.clone(),
            actual: observation.descriptor.field_id.clone(),
        });
    }
    if descriptor.field_id != adapter.output_field_id {
        return Err(StateBoundaryError::AssimilationOutputFieldMismatch {
            expected: adapter.output_field_id.clone(),
            actual: descriptor.field_id.clone(),
        });
    }
    Ok(PrognosticField {
        descriptor,
        value,
        initialization: InitializationProvenance::AssimilatedObservation {
            adapter_id: adapter.adapter_id.clone(),
            adapter_version: adapter.semantic_version.clone(),
            source_id: observation.provenance.source_id.clone(),
            source_artifact_digest: observation.provenance.artifact_digest.clone(),
            source_field_id: observation.descriptor.field_id.clone(),
        },
    })
}

impl<T> ExperimentalRepresentation<T> {
    pub fn new(
        representation_id: impl Into<String>,
        source_field_ids: Vec<String>,
        value: T,
    ) -> Result<Self, StateBoundaryError> {
        let representation_id = representation_id.into();
        if representation_id.trim().is_empty() {
            return Err(StateBoundaryError::EmptyFieldId);
        }
        if source_field_ids.is_empty() {
            return Err(StateBoundaryError::EmptyRepresentationSources);
        }
        if source_field_ids.iter().any(|item| item.trim().is_empty()) {
            return Err(StateBoundaryError::EmptyFieldId);
        }
        let mut unique = BTreeSet::new();
        for field_id in &source_field_ids {
            if !unique.insert(field_id.as_str()) {
                return Err(StateBoundaryError::DuplicateSourceFieldId {
                    field_id: field_id.clone(),
                });
            }
        }
        Ok(Self {
            representation_id,
            source_field_ids,
            value,
        })
    }
}
