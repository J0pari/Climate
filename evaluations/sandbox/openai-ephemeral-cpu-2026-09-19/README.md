# OpenAI ephemeral CPU campaign — 2026-09-19

Repository revision: `27689bf3b27e8980a904cb70075091e01222e759`

Classification: **exploratory, non-promotional evaluation**.

Execution venue: OpenAI ephemeral CPU sandbox with 5 CPU cores, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, and scikit-learn 1.8.0 where used. The sandbox had no GPU, public outbound package/network access, CUE, Rust, PyDMD, or datafold. Missing capabilities were left unavailable rather than replaced by local implementations under the same identity.

These runs are independent replications and robustness checks. They are not canonical Climate run receipts, are not CUE-vetted, and do not enter `evidence/registry.json`.

## Registered EBM replications

All six registered EBM experiments whose declared methods were executable without PyDMD satisfied their registered structural/numerical checks in this environment.

- `physics.two_layer_ebm.forcing_protocols.v1`: constant-equivalence maximum state error `4.441e-16 K`; maximum energy-budget residual `1.332e-15 W m^-2`.
- `physics.two_layer_ebm.parameter_identifiability.v1`: equilibrium rank `1`, transient rank `4`, rank gain `3`; transient derivative step-consistency `6.084e-10`.
- `multirepresentation.ebm_stochastic_statistics.v1`: full-state covariance and lag-covariance errors `0`; maximum total-budget residual `2.776e-16 W m^-2`; redundant-vs-surface covariance-error delta `0`.
- `multirepresentation.ebm_regime_feedback.v1`: raw confirmation error `0.005420`; gated error `3.704e-16`; swapped-label error `0.025687`; observed minimum regime margin `0.753324 K`.
- `multirepresentation.ebm_forced_ood.v1`: full-state confirmation error `2.063e-16`; surface-scalar error `0.050960`; closure gap `0.050960`. An additional 4,096-sample confirmation-box stress test retained `4.679e-16` relative error for the full-state map.
- `multirepresentation.ebm_observation_degradation.v1`: clean-control forced-prediction error `2.063e-16`; noisy-full-versus-surface prediction gap `0.322001`; redundant structural-rank gain `0`.

## Exploratory robustness sweeps

These sweeps were deliberately outside the preregistered experiment versions. They may guide future preregistration but do not alter registered thresholds retroactively.

- Forced OOD, 256 discovery seeds: no full-state errors exceeded `1e-12`; no surface-scalar confirmation error fell below the registered `0.02` lower bound. Surface confirmation errors ranged `0.02231–0.07761`.
- Regime feedback, 128 independent seed quartets: no raw, gated, swapped-label, or regime-margin threshold failures. Raw confirmation errors ranged `0.005020–0.005826`.
- Stochastic control, 128 seeds: maximum budget residual remained at floating-point scale; surface-only covariance loss remained materially nonzero for every tested seed.
- Parameter log-step scan: equilibrium rank `1` and transient rank `4` for all 13 tested steps from `1e-2` through `1e-7`.
- Observation-noise dose response, 32 seeds per level: noisy-full forced-prediction error rose smoothly from numerical zero at zero noise to about `0.153` mean at `0.2 K` noise.

## Independent cross-language oracle

A GNU Fortran RK4 implementation was compared with the Python/SciPy exact affine matrix-exponential forcing solution. Halving the timestep reduced error by approximately 16x, consistent with fourth-order convergence. At `h=0.0125 yr`, the maximum state disagreement across yearly protocol samples was `6.928e-13 K`.

## Active structural-world node

For `multirepresentation.structure_baselines`, the sandbox executed the authoritative seven-world semantics, the 255-permutation mutual-information selector, available PCA/CCA/spectral/factor-analysis baselines, and the matched-information probe for raw/PCA/CCA/factor representations.

All discovery and confirmation selector decisions were calibrated on this realization.

A follow-up exploratory seed-offset robustness sweep evaluated eight independent realizations of each of the seven worlds (56 world realizations total) with the same 255-permutation selector policy. All 56 decisions were calibrated. The product and independent-null worlds abstained in every realization; every world with an authoritative shared target supported a shared coordinate in every realization. See `structural-selector-robustness.json` for per-world p-value and statistic ranges. `datafold`/JSF and PyDMD temporal state-space remained unavailable and were not substituted.

## Independent automatic-differentiation geometry witness

A JAX 0.9.0.1 `jacfwd` route independently differentiated all five analytic geometry fixtures at 64 interior points each and compared first/second metric derivatives with exact SymPy derivatives. In FP64, the worst first-derivative, second-derivative, and scalar-curvature errors were `2.13e-14`, `1.71e-13`, and `3.55e-15`. In FP32 they rose to `1.02e-5`, `9.68e-5`, and `2.62e-6`, respectively. This is an exploratory maintained-library AD witness, not yet a canonical repository execution route.

## Resource accounting

No GitHub Codespaces compute was consumed by this campaign.

See `campaign.json` in this directory for the machine-readable compact record.
