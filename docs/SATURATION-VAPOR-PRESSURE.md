# Saturation-vapor-pressure semantics

This document defines the canonical phase-explicit equilibrium-vapor-pressure provider used by the portable Fortran surface.

## Authority and scope

`src/fortran/saturation_vapor_pressure.f90` implements the liquid-water and hexagonal-ice parameterizations from:

D. M. Murphy and T. Koop (2005), *Review of the vapour pressures of ice and supercooled water for atmospheric applications*, Quarterly Journal of the Royal Meteorological Society 131, 1539–1565, doi:10.1256/qj.04.94.

The provider returns equilibrium water-vapor partial pressure in pascals from temperature in kelvin and an explicit phase selector. It does not infer phase from temperature.

This authority is local to Climate kernels and reference comparisons. When an external climate model, reanalysis, or preprocessing system uses its own saturation/phase formulation, Climate must record and preserve that native thermodynamic definition for the run rather than silently substituting this provider. Cross-model comparisons may use this implementation as an independent oracle only when the compared formula/domain is explicitly matched.

## Liquid-water branch

The liquid-water expression is the Murphy–Koop supercooled-liquid parameterization

\[
\ln e_w =
54.842763-\frac{6763.22}{T}-4.210\ln T+0.000367T
+\tanh(0.0415(T-218.8))
\left(53.878-\frac{1331.22}{T}-9.44523\ln T+0.014025T\right),
\]

with `e_w` in Pa and the published validity interval

\[
123\ \mathrm{K} < T < 332\ \mathrm{K}.
\]

The interval is enforced as an open interval. Calls at or outside either published boundary fail closed.

## Ice branch

The hexagonal-ice expression is

\[
\ln e_i =
9.550426-\frac{5723.265}{T}+3.53068\ln T-0.00728332T.
\]

Murphy and Koop document this expression for `T > 110 K`. The canonical Climate API adds a deliberate upper policy bound at the water triple point, `273.16 K`. The reason is semantic rather than numerical: a caller requesting ordinary stable hexagonal ice above the triple point is making a phase-policy decision that must not be silently embedded in a low-level equilibrium-pressure helper.

This additional bound does not claim that the analytic expression suddenly becomes numerically undefined above `273.16 K`; it prevents the canonical stable-ice interface from silently extending into a metastable/non-stable phase regime.

## No automatic phase switching

There is intentionally no `auto` phase mode.

At temperatures below freezing, supercooled liquid water and ice have different equilibrium vapor pressures. Automatically selecting a branch from temperature would erase a scientifically material state distinction. Higher layers that know the condensate phase may choose the appropriate branch explicitly.

## Relationship to moist-vapor algebra

`src/fortran/moist_vapor_algebra.f90` owns exact ideal-gas transforms among vapor pressure, mixing ratio, specific humidity, virtual temperature, and vapor-only density. It does not own saturation physics.

The composition boundary is therefore:

`temperature + explicit phase -> saturation vapor pressure -> vapor algebra`

For example, a saturation mixing ratio may be obtained by computing `e_s(T, phase)` here and then passing that pressure to `compute_mixing_ratio_from_vapor_pressure`. The latter still requires `e_s < total_pressure`; no saturation provider may bypass that domain requirement.

## Executable witnesses

The Fortran witnesses require:

- direct decimal fixtures from independent evaluation of the published equations;
- monotonic pressure increase over representative temperature intervals;
- distinct liquid-water and ice pressures under supercooled conditions;
- close agreement of the two parameterizations at the water triple point;
- exact fail-closed enforcement of published/open validity boundaries;
- rejection of an unknown phase selector;
- rejection of non-finite temperatures;
- and round-trip composition through the canonical vapor-pressure/mixing-ratio algebra.

These tests establish transcription, domain, and software behavior. They do not validate a cloud microphysics scheme, phase partition, supersaturation treatment, latent-heat closure, or prognostic atmospheric model.
