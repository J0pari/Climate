# Oscillation and mode diagnostics blueprint

Status: **research specification**.

Climate oscillations and recurrent modes should be represented by small, independently testable diagnostics rather than by one omnibus monitor. Index definitions, spectral transforms, phase representations, mode extraction, and forecast verification are distinct responsibilities and may mature independently.

## Capability partitions

```text
oscillation/data_contracts
  SST/SLP/wind/OLR input identities, coordinates, masks, anomaly baselines

oscillation/indices
  narrowly defined observational indices such as Niño3.4 or DMI

oscillation/spectral
  spectra, coherence, phase, and time-frequency transforms with independent references

oscillation/modes
  EOF/complex-EOF/state-space/Koopman representations

oscillation/coupling
  cross-mode phase, amplitude, resonance, and directional-coupling diagnostics

oscillation/forecast_evaluation
  hindcast-only skill metrics from actual forecast/observation pairs
```

## Representation role

Oscillation products may participate in the multirepresentation climate program as observation maps, reduced coordinates, local charts, or relational features. Their role must be declared rather than assuming that an index or spectral coefficient is automatically a physical state coordinate.

Useful questions include:

- whether established climate indices align with dynamically coherent directions;
- whether spectral or Koopman coordinates provide useful slow/oscillatory factors;
- whether phase-amplitude structure adds information beyond ordinary spectra;
- whether oriented or multicomponent mode interactions justify Clifford/geometric-algebra representations;
- whether mode relationships are stable across datasets, epochs, and spatial resolutions;
- whether integrating oscillatory representations with thermodynamic, dynamical, or information-geometric views improves trajectory or regime representation.

## Non-negotiable semantics

- Observational index definitions must identify source region, anomaly baseline, temporal filtering, standardization, and data product.
- Spectral and time-frequency methods require synthetic known-frequency fixtures and explicit null models.
- Phase and coherence quantities must declare sign, lag, unwrap, smoothing, and endpoint conventions.
- ENSO/PDO/AMV/NAO/IOD/MJO classifications must separate index computation from physical interpretation.
- Missing required data produces an explicit unavailable/error result unless the experiment explicitly studies an imputation rule.
- Forecast skill requires actual hindcast/prediction/observation pairs and a declared verification metric.
- A mode decomposition must state whether it targets variance, predictability, dynamical closure, oscillatory coherence, or another objective; those are not interchangeable.

## Minimum verification

Before a diagnostic is treated as more than a prototype, require as applicable:

1. deterministic index fixtures with analytically checkable regional averages;
2. sinusoid, multi-frequency, chirp, and AR(1) spectral fixtures;
3. phase/coherence fixtures with known lag and amplitude relation;
4. missing-data, irregular-time, and calendar tests;
5. comparison against trusted external implementations for selected established metrics;
6. mode-reconstruction and orthogonality/biorthogonality witnesses appropriate to the method;
7. hindcast skill metrics tested on planted prediction/observation arrays where expected RMSE, correlation, Brier, or proper-score quantities are known;
8. explicit refusal of unsupported interpretations under `docs/SEMANTIC-SANITATION.md`.

## Scientific comparison

No single oscillation representation is presumed canonical for every task. Compare candidates against simpler alternatives with matched information access:

- standard climate indices;
- Fourier/Welch/multitaper spectra;
- Hilbert or wavelet phase;
- EOF/complex EOF;
- state-space models;
- DMD/Koopman methods;
- graph or coherence networks;
- learned latent sequence representations.

Clifford, ultrametric, geometric, or other unusual representations should be evaluated on the specific structure they claim to expose rather than on generic reconstruction alone.

## Resource shape

Reference index, spectral, and mode verification should remain portable CPU work wherever practical. GPU acceleration is justified only when profiling demonstrates a real high-volume bottleneck and must preserve the same declared semantics. Large observational or hindcast evaluation belongs to the large-data execution class. Commons integration should consume standard experiment/run/artifact records rather than requiring a special oscillation scheduler.
