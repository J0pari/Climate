//! Typed boundaries between observations, prognostic fields, and experimental representations.
//!
//! This module owns state-role, availability, and initialization provenance only.
//! Domain kernels retain ownership of numerical validity, discretization, units
//! interpretation, and field-specific invariants. Values remain generic so a
//! scalar, array, or domain-owned field type can cross the same semantic boundary
//! without introducing a repository-wide state container.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldDescriptor {
    pub field_id: String,
    pub units: String,
    pub coordinates: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Availability<T> {
    Present { value: T },
    Missing { reason: String },
    Unavailable { reason: String },
}

impl<T> Availability<T> {
    pub fn present(value: T) -> Self {
        Self::Present { value }
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
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InitializationProvenance {
    ExplicitModelInput,
    Restart {
        artifact_digest: String,
    },
    AssimilatedObservation {
        adapter_id: String,
        source_id: String,
        source_artifact_digest: String,
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
    adapter_id: impl Into<String>,
) -> Result<PrognosticField<T>, StateBoundaryError> {
    validate_descriptor(&descriptor)?;
    if observation.availability.value().is_none() {
        return Err(StateBoundaryError::ObservationNotPresent);
    }
    let adapter_id = adapter_id.into();
    if adapter_id.trim().is_empty() {
        return Err(StateBoundaryError::EmptyAdapterId);
    }
    Ok(PrognosticField {
        descriptor,
        value,
        initialization: InitializationProvenance::AssimilatedObservation {
            adapter_id,
            source_id: observation.provenance.source_id.clone(),
            source_artifact_digest: observation.provenance.artifact_digest.clone(),
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
        if source_field_ids.iter().any(|item| item.trim().is_empty()) {
            return Err(StateBoundaryError::EmptyFieldId);
        }
        Ok(Self {
            representation_id,
            source_field_ids,
            value,
        })
    }
}
