# Oscillation monitor blueprint

Status: **capability unavailable / legacy executable prototype removed**.

The previous `climate_oscillation_monitor.f90` attempted to cover ENSO, PDO, AMO/AMV, NAO, IOD, MJO, wavelet/spectral diagnostics, and forecast skill in one executable module. It mixed useful domain notes with arbitrary/fake skill formulas, unvalidated constants, incomplete numerical transforms, and a test program that consumed uninitialized arrays and was explicitly documented as meaningless/crashing.

That implementation was removed from the live execution path rather than preserved as a successful-looking prototype.

## Intended capability

A future oscillation subsystem should provide small independently testable diagnostics rather than one omnibus monitor.

Suggested partitions:

```text
oscillation/data_contracts
  SST/SLP/wind/OLR input identities, coordinates, masks, anomaly baselines

oscillation/indices
  narrowly defined observational indices (Niño3.4, DMI, etc.)

oscillation/spectral
  spectra, coherence, phase, wavelet transforms with independent references

oscillation/modes
  EOF/complex-EOF/state-space representations

oscillation/forecast_evaluation
  hindcast-only skill metrics from actual forecast/observation pairs
```

## Non-negotiable semantics

- No forecast skill is computed from hand-chosen decay curves and returned under a real skill metric name.
- Observational index definitions must identify source region, anomaly baseline, temporal filtering, standardization, and data product.
- Spectral/wavelet methods require synthetic known-frequency fixtures and red-noise/null controls.
- Any MJO/ENSO/PDO/etc. classifier must separate index computation from physical interpretation.
- Missing required data produces an explicit unavailable/error result, not a climatological/default substitute unless the experiment explicitly studies that imputation.
- Forecast skill requires actual hindcast/prediction/observation pairs and a declared verification metric.

## Minimum reintroduction tests

Before an implementation replaces the unavailable stub:

1. tiny deterministic index fixtures with analytically checkable regional averages;
2. sinusoid + multi-frequency + AR(1) spectral fixtures;
3. phase/coherence fixtures with known lag;
4. missing-data and irregular-time tests;
5. comparison against a trusted external implementation for selected metrics;
6. hindcast skill metrics tested on planted prediction/observation arrays where expected RMSE/correlation/Brier quantities are known;
7. no S3/S4 semantic hazards under `docs/SEMANTIC-SANITATION.md`.

## Resource shape

Most reference/index/spectral verification should be `R1_portable_cpu`. GPU acceleration is optional and justified only after profiling shows a real bottleneck. Large observational/hindcast evaluation is `R5_large_data`. Commons integration should consume experiment/run/artifact records rather than a special oscillation scheduler.

## Legacy recovery

The removed prototype remains available in Git history before the semantic-sanitation commits. It should be consulted as a source of ideas/data-source notes only, not copied back wholesale.
