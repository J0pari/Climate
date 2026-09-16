# Spectral and time-frequency diagnostics blueprint

Status: **capability unavailable / legacy omnibus implementation removed**.

The previous `climate_spectral_analysis.f90` exposed EEMD/CEEMDAN, CWT, synchrosqueezing, SSA, multitaper spectra, Hilbert-Huang analysis, Hilbert phase, wavelet coherence, MEM/Burg spectra, Lomb-Scargle and related methods under one module. Several advertised routines were empty, returned zero arrays, or had unresolved implementation/interface errors. Keeping those procedures callable under real algorithm names was a semantic hazard.

The live source is therefore an unavailable stub. Git history preserves the exploratory implementation for reference only.

## Target partitions

Reintroduce methods independently rather than rebuilding one omnibus module:

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

Each partition gets its own `MethodDescriptor`, fixtures, baselines, and maturity.

## Minimum verification by family

### FFT / ordinary spectra

- planted single- and multi-sinusoid frequency/amplitude fixtures;
- Parseval/power normalization checks;
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
- no zero-filled IMF placeholder outputs.

## Scientific use

Spectral methods are established families but individual implementations are not self-validating. Climate-specific claims require data preprocessing, null models, dependence-aware uncertainty, and baseline comparisons. A numerically verified transform may reach `verified`; it does not make an oscillation/tipping interpretation `validated`.

## Acceleration

Reference verification is `R1_portable_cpu`. GPU acceleration should be introduced only for demonstrated high-volume workloads, behind the same method semantics and with CPU/GPU differential witnesses. Do not make CUDA the reference definition of a transform.
