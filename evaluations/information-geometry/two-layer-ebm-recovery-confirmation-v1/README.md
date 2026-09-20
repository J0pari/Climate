# Two-layer EBM information-geometry recovery confirmation v1

This directory stores the repository-persisted result of the preregistered confirmation experiment \`information_geometry.two_layer_ebm.recovery_confirmation.v1\`.

Execution identity:

- source commit: \`2d371769aaed9a554e51b51a13919d9b56e2ccfc\`;
- Python 3.12;
- NumPy 2.5.3;
- SciPy 1.18.1;
- 4 fresh synthetic observation seeds crossed with 4 frozen log-parameter starts = 16 matched trials;
- discovery forcing: \`held_out_step\` and \`held_out_ramp\`;
- confirmation forcing: \`overshoot_reversal\` and \`sign_reversal\`.

The exact machine result is \`result.json\`.

## Preregistered outcome

The local-Fisher natural-gradient candidate and the primary SciPy Jacobian-scaled TRF comparator both converged in 16/16 trials. Their median held-out negative log likelihood and median log-parameter RMSE were equivalent within the frozen \`1e-6\` tolerances.

Natural gradient:

- median held-out NLL per observation: \`-1.3483276076474928\`;
- median log-parameter RMSE: \`0.27874662392030647\`;
- median residual evaluations: \`15.5\`.

TRF:

- median held-out NLL per observation: \`-1.3483277096094075\`;
- median log-parameter RMSE: \`0.2787465207637703\`;
- median residual evaluations: \`10.5\`.

The candidate therefore did not satisfy the preregistered rule for a distinct information-geometry optimization advantage. The local Fisher step also agreed with ordinary Gauss-Newton to a maximum relative difference of \`2.5341573334591624e-13\`, inside the frozen \`1e-10\` threshold.

The frozen-initial-Fisher negative control converged in 0/16 trials, while SciPy L-BFGS-B converged in 16/16.

## Authority boundary

This is a repository-owned confirmation evaluation, not an entry in \`evidence/registry.json\`. It completes the planning obligation to perform the explicit likelihood/recovery comparison, but it does not by itself change the maturity of \`climate.information_geometry.optimization\`.

The result is attacking with respect to any interpretation that Fisher geometry provides a unique optimization advantage on this fixed-variance Gaussian two-layer EBM recovery task. It does not rule out advantages for different likelihoods, parameterizations, approximations, non-Gaussian errors, priors, constraints, or model classes.
