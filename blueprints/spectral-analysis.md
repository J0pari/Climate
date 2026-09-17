# Spectral and time-frequency diagnostics blueprint

Status: **research specification**.

Spectral and time-frequency methods should be introduced as independently specified capabilities rather than as one omnibus transform surface. Each method family needs its own mathematical conventions, fixtures, external references, and maturity.

## Capability partitions

```text
spectral/fft_reference
spectral/welch
spectral/multitaper
spectral/lomb_scargle
spectral/wavelet
spectral/wavelet_coherence
spectral/hilbert
spectral/ssa
spectral/emd
```

Each partition gets its own method identity, fixtures, baselines, and evidence scope.

## Mathematical conventions

Every spectral or time-frequency method must declare the conventions that materially affect interpretation:

- sampling interval and time coordinate;
- detrending/anomaly policy;
- normalization and units of amplitude/power;
- one-sided versus two-sided spectra;
- window and taper normalization;
- frequency/scale convention;
- endpoint and padding policy;
- phase sign and unwrap convention;
- smoothing and cone-of-influence semantics where relevant;
- null/noise model used for significance or uncertainty.

These choices are part of the method contract, not plotting details.

## Minimum verification by family

### FFT / ordinary spectra

- planted single- and multi-sinusoid frequency/amplitude fixtures;
- Parseval/power-normalization checks;
- odd/even lengths and non-power-of-two sizes;
- window normalization and leakage witnesses;
- comparison with a trusted CPU library.

### Multitaper

- verified DPSS generation or trusted library binding;
- time-bandwidth/taper-count constraints;
- eigenspectrum averaging and normalization tests;
- known colored-noise/null examples.

### Lomb-Scargle

- irregularly sampled sinusoid recovery;
- comparison with a trusted statistical/scientific implementation;
- explicit normalization/significance convention;
- missing/duplicate time handling.

### Wavelets / coherence

- known chirp and transient fixtures;
- scale-frequency convention and cone-of-influence tests;
- phase-lag/coherence planted pairs;
- red-noise/null controls;
- smoothing semantics pinned for coherence.

### Hilbert / instantaneous frequency

- analytic-signal comparison on narrowband fixtures;
- phase unwrapping before differencing;
- sampling interval explicit rather than assumed;
- endpoint policy and negative-frequency handling tested.

### SSA

- actual Hankel/trajectory construction;
- eigendecomposition/reconstruction identities;
- planted trend + oscillation decomposition;
- window/component constraints;
- comparison with an independent reference.

### EMD/EEMD/CEEMDAN

- actual extrema/envelope/sifting implementation or verified external library;
- stopping criterion explicitly versioned;
- noise ensemble seed/provenance;
- mode-mixing and end-effect fixtures;
- no zero-filled or otherwise fabricated IMF outputs.

## Role in climate representation

Spectral products may serve as diagnostics, reduced coordinates, observation maps, kernels, or factors in the multirepresentation climate-state program. Their scientific role must be declared for each experiment.

Important questions include:

- whether a spectral coordinate captures a dynamically coherent process rather than only high variance;
- whether phase or coherence structure is stable across preprocessing choices and datasets;
- whether time-frequency representations improve local state geometry or regime separation beyond simpler summaries;
- whether spectral/Koopman coordinates provide useful slow, oscillatory, or approximately invariant factors;
- whether multicomponent mode interactions require richer algebraic representations than ordinary complex amplitudes.

A numerically correct transform is not automatically a useful climate coordinate.

## Library boundary

Mature generic transforms should use maintained libraries in production when their semantics match the declared method. Transparent local implementations are justified as small differential/reference oracles, not as a reason to build a bespoke general-purpose signal-processing stack.

For ordinary production FFTs, FFTW is the preferred CPU boundary and cuFFT the preferred GPU boundary unless a concrete requirement demonstrates a better fit. Direct DFT remains useful as a deterministic small-size oracle.

## Scientific use

Climate-specific claims require explicit preprocessing, null models, dependence-aware uncertainty, and baseline comparisons. A numerically verified transform may reach `verified`; it does not make an oscillation, regime, teleconnection, or tipping interpretation `validated`.

## Acceleration

Reference verification should remain portable CPU work. GPU acceleration should be introduced only for demonstrated high-volume workloads, behind the same method semantics and with CPU/GPU differential witnesses. CUDA does not define the mathematical reference behavior of a transform.
