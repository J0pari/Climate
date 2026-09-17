use std::fs;
use std::path::PathBuf;

use climate_geometric_framework::ebm_observation_geometry::{
    two_layer_state_fisher, TwoLayerObservation, TwoLayerObservationChannel,
};
use serde_json::Value;

fn fixture_json(path: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
    let text = fs::read_to_string(path).expect("fixture must be readable");
    serde_json::from_str(&text).expect("fixture must be valid JSON")
}

fn number(value: &Value, pointer: &str) -> f64 {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("missing numeric fixture value at {pointer}"))
}

fn channel(observation: TwoLayerObservation, sigma: f64) -> TwoLayerObservationChannel {
    TwoLayerObservationChannel::new(observation, sigma)
}

fn approx_eq(left: f64, right: f64, relative: f64) {
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(
        (left - right).abs() <= relative * scale,
        "left={left:e}, right={right:e}, relative={relative:e}"
    );
}

#[test]
fn observation_control_fixture_exposes_information_complementarity() {
    let ebm = fixture_json("fixtures/physics/two-layer-ebm-geoffroy-mean-v1.json");
    let observations =
        fixture_json("fixtures/physics/two-layer-ebm-observation-control-v1.json");

    assert_eq!(
        observations["provenance"]["kind"].as_str(),
        Some("synthetic_structural_control")
    );
    assert_eq!(
        observations["ebm_fixture_id"].as_str(),
        ebm["fixture_id"].as_str()
    );

    let lambda = number(&ebm, "/parameters/climate_feedback_w_m2_k");
    let gamma = number(&ebm, "/parameters/ocean_heat_exchange_w_m2_k");
    let sigma_surface = number(
        &observations,
        "/channels/surface_temperature/standard_deviation",
    );
    let sigma_toa = number(&observations, "/channels/toa_imbalance/standard_deviation");
    let sigma_ocean = number(
        &observations,
        "/channels/ocean_heat_uptake/standard_deviation",
    );

    let surface = two_layer_state_fisher(
        lambda,
        gamma,
        &[channel(
            TwoLayerObservation::SurfaceTemperature,
            sigma_surface,
        )],
    )
    .unwrap();
    let surface_toa = two_layer_state_fisher(
        lambda,
        gamma,
        &[
            channel(TwoLayerObservation::SurfaceTemperature, sigma_surface),
            channel(TwoLayerObservation::ToaImbalance, sigma_toa),
        ],
    )
    .unwrap();
    let surface_ocean = two_layer_state_fisher(
        lambda,
        gamma,
        &[
            channel(TwoLayerObservation::SurfaceTemperature, sigma_surface),
            channel(TwoLayerObservation::OceanHeatUptake, sigma_ocean),
        ],
    )
    .unwrap();
    let all = two_layer_state_fisher(
        lambda,
        gamma,
        &[
            channel(TwoLayerObservation::SurfaceTemperature, sigma_surface),
            channel(TwoLayerObservation::ToaImbalance, sigma_toa),
            channel(TwoLayerObservation::OceanHeatUptake, sigma_ocean),
        ],
    )
    .unwrap();

    assert_eq!(surface.parameter_nullity, 1);
    assert_eq!(surface_toa.parameter_nullity, 1);
    assert_eq!(surface_ocean.parameter_nullity, 0);
    assert_eq!(all.parameter_nullity, 0);

    let surface_precision = 1.0 / sigma_surface.powi(2);
    let toa_precision = lambda.powi(2) / sigma_toa.powi(2);
    let ocean_precision = gamma.powi(2) / sigma_ocean.powi(2);

    approx_eq(surface.fisher[(0, 0)], surface_precision, 1e-13);
    approx_eq(
        surface_toa.fisher[(0, 0)],
        surface_precision + toa_precision,
        1e-13,
    );
    approx_eq(surface_toa.fisher[(1, 1)], 0.0, 1e-13);

    approx_eq(
        surface_ocean.fisher[(0, 0)],
        surface_precision + ocean_precision,
        1e-13,
    );
    approx_eq(
        surface_ocean.fisher[(0, 1)],
        -ocean_precision,
        1e-13,
    );
    approx_eq(
        surface_ocean.fisher[(1, 1)],
        ocean_precision,
        1e-13,
    );

    approx_eq(
        all.fisher[(0, 0)],
        surface_precision + toa_precision + ocean_precision,
        1e-13,
    );
    approx_eq(all.fisher[(0, 1)], -ocean_precision, 1e-13);
    approx_eq(all.fisher[(1, 1)], ocean_precision, 1e-13);
}

#[test]
fn reducing_one_channel_noise_changes_precision_not_observation_span() {
    let ebm = fixture_json("fixtures/physics/two-layer-ebm-geoffroy-mean-v1.json");
    let lambda = number(&ebm, "/parameters/climate_feedback_w_m2_k");
    let gamma = number(&ebm, "/parameters/ocean_heat_exchange_w_m2_k");

    let coarse = two_layer_state_fisher(
        lambda,
        gamma,
        &[
            channel(TwoLayerObservation::SurfaceTemperature, 0.1),
            channel(TwoLayerObservation::ToaImbalance, 0.2),
        ],
    )
    .unwrap();
    let precise_toa = two_layer_state_fisher(
        lambda,
        gamma,
        &[
            channel(TwoLayerObservation::SurfaceTemperature, 0.1),
            channel(TwoLayerObservation::ToaImbalance, 0.02),
        ],
    )
    .unwrap();

    assert_eq!(coarse.parameter_nullity, 1);
    assert_eq!(precise_toa.parameter_nullity, 1);
    assert!(precise_toa.fisher[(0, 0)] > coarse.fisher[(0, 0)]);
    approx_eq(precise_toa.fisher[(1, 1)], 0.0, 1e-13);
}
