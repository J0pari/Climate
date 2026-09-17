use climate_geometric_framework::physical_state::{
    assimilated_prognostic_field, Availability, ExperimentalRepresentation, FieldDescriptor,
    InitializationProvenance, ObservationProvenance, ObservedField, PrognosticField,
    StateBoundaryError,
};

const NCEI_DIGEST: &str =
    "sha256:06e00068f7de97d76b8bc523d5acb22ec7b21631adfec0e0c1a284faf1ac1a46";

fn temperature_descriptor() -> FieldDescriptor {
    FieldDescriptor {
        field_id: "air_temperature".to_string(),
        units: "K".to_string(),
        coordinates: "time station".to_string(),
    }
}

fn observed_temperature(availability: Availability<f64>) -> ObservedField<f64> {
    ObservedField::new(
        temperature_descriptor(),
        availability,
        ObservationProvenance::new("ncei.ghcnd.v3", NCEI_DIGEST).unwrap(),
    )
    .unwrap()
}

#[test]
fn zero_missing_and_unavailable_are_distinct_states() {
    let zero = Availability::present(0.0_f64);
    let missing = Availability::<f64>::Missing {
        reason: "provider missing value".to_string(),
    };
    let unavailable = Availability::<f64>::Unavailable {
        reason: "field was not requested".to_string(),
    };

    assert_eq!(zero.value(), Some(&0.0));
    assert_eq!(missing.value(), None);
    assert_eq!(unavailable.value(), None);

    let zero_json = serde_json::to_value(&zero).unwrap();
    let missing_json = serde_json::to_value(&missing).unwrap();
    let unavailable_json = serde_json::to_value(&unavailable).unwrap();
    assert_eq!(zero_json["status"], "present");
    assert_eq!(zero_json["value"], 0.0);
    assert_eq!(missing_json["status"], "missing");
    assert!(missing_json.get("value").is_none());
    assert_eq!(unavailable_json["status"], "unavailable");
    assert!(unavailable_json.get("value").is_none());
}

#[test]
fn observation_requires_dataset_artifact_identity() {
    assert_eq!(
        ObservationProvenance::new("ncei.ghcnd.v3", "not-a-digest"),
        Err(StateBoundaryError::InvalidArtifactDigest)
    );
    assert_eq!(
        ObservationProvenance::new("", NCEI_DIGEST),
        Err(StateBoundaryError::EmptySourceId)
    );
}

#[test]
fn assimilation_is_explicit_and_preserves_observation_provenance() {
    let observation = observed_temperature(Availability::present(281.45));
    let prognostic = assimilated_prognostic_field(
        FieldDescriptor {
            field_id: "surface_air_temperature".to_string(),
            units: "K".to_string(),
            coordinates: "model_grid".to_string(),
        },
        vec![281.1, 281.8],
        &observation,
        "assimilation.station_to_model_grid.v1",
    )
    .unwrap();

    assert_eq!(prognostic.value, vec![281.1, 281.8]);
    assert_eq!(
        prognostic.initialization,
        InitializationProvenance::AssimilatedObservation {
            adapter_id: "assimilation.station_to_model_grid.v1".to_string(),
            source_id: "ncei.ghcnd.v3".to_string(),
            source_artifact_digest: NCEI_DIGEST.to_string(),
        }
    );
}

#[test]
fn missing_observation_cannot_initialize_prognostic_state() {
    let observation = observed_temperature(Availability::Missing {
        reason: "no report".to_string(),
    });
    let result = assimilated_prognostic_field(
        temperature_descriptor(),
        280.0,
        &observation,
        "assimilation.station_scalar.v1",
    );
    assert_eq!(result, Err(StateBoundaryError::ObservationNotPresent));
}

#[test]
fn explicit_and_restart_initialization_remain_distinct() {
    let explicit = PrognosticField::explicit(temperature_descriptor(), 280.0).unwrap();
    assert_eq!(
        explicit.initialization,
        InitializationProvenance::ExplicitModelInput
    );

    let restart = PrognosticField::from_restart(
        temperature_descriptor(),
        280.0,
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    .unwrap();
    assert_eq!(
        restart.initialization,
        InitializationProvenance::Restart {
            artifact_digest:
                "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
        }
    );
}

#[test]
fn experimental_representation_is_a_separate_typed_artifact() {
    let representation = ExperimentalRepresentation::new(
        "geometry.latent_coordinate.v1",
        vec!["surface_air_temperature".to_string()],
        vec![0.25, -0.1],
    )
    .unwrap();
    assert_eq!(representation.source_field_ids.len(), 1);
    assert_eq!(representation.value, vec![0.25, -0.1]);
}
