"""Narrow library-backed spectral diagnostics with explicit physical semantics.

This module delegates the one-sided periodogram to SciPy.  It deliberately
rejects missing/non-finite samples rather than interpolating them and does not
construct named climate indices, forecast skill, or significance claims.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal


PERIODOGRAM_BACKEND = "scipy.signal.periodogram"
PERIODOGRAM_WINDOW = "boxcar"
PERIODOGRAM_SCALING = "density"
MISSINGNESS_POLICY = "reject_nonfinite"


@dataclass(frozen=True)
class PeriodogramReport:
    frequency_hz: np.ndarray
    power_spectral_density: np.ndarray
    sample_interval_seconds: float
    signal_unit: str
    detrend: str
    backend: str = PERIODOGRAM_BACKEND
    window: str = PERIODOGRAM_WINDOW
    scaling: str = PERIODOGRAM_SCALING
    missingness_policy: str = MISSINGNESS_POLICY
    frequency_unit: str = "Hz"

    def __post_init__(self) -> None:
        frequency = np.asarray(self.frequency_hz, dtype=np.float64)
        density = np.asarray(self.power_spectral_density, dtype=np.float64)
        if frequency.ndim != 1 or density.ndim != 1 or frequency.shape != density.shape:
            raise ValueError("frequency and power spectral density must be matching vectors")
        if frequency.size < 2:
            raise ValueError("periodogram report requires at least two frequency bins")
        if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(density)):
            raise ValueError("periodogram report must be finite")
        if np.any(frequency < 0.0) or np.any(np.diff(frequency) <= 0.0):
            raise ValueError("periodogram frequencies must be strictly increasing and non-negative")
        if np.any(density < 0.0):
            raise ValueError("power spectral density must be non-negative")
        if not np.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0.0:
            raise ValueError("sample interval must be finite and positive")
        if not self.signal_unit.strip():
            raise ValueError("signal unit must be explicit")
        if self.detrend not in {"none", "constant"}:
            raise ValueError("unsupported detrend policy")
        frequency = np.array(frequency, copy=True)
        density = np.array(density, copy=True)
        frequency.setflags(write=False)
        density.setflags(write=False)
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "power_spectral_density", density)

    @property
    def power_spectral_density_unit(self) -> str:
        return f"({self.signal_unit})^2/Hz"

    def integrated_power(self) -> float:
        spacing = np.diff(self.frequency_hz)
        representative = float(np.mean(spacing))
        tolerance = (
            16.0
            * np.finfo(np.float64).eps
            * max(abs(representative), float(np.max(np.abs(self.frequency_hz))))
        )
        if np.max(np.abs(spacing - representative)) > tolerance:
            raise ValueError("periodogram frequency grid is not uniformly spaced")
        return float(np.sum(self.power_spectral_density) * representative)


def one_sided_periodogram(
    samples: np.ndarray,
    *,
    sample_interval_seconds: float,
    signal_unit: str,
    detrend: str = "constant",
) -> PeriodogramReport:
    """Compute a one-sided PSD using SciPy with explicit missingness semantics."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("periodogram requires at least three one-dimensional samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("periodogram rejects non-finite or missing samples")
    if not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0.0:
        raise ValueError("sample interval must be finite and positive")
    if not isinstance(signal_unit, str) or not signal_unit.strip():
        raise ValueError("signal unit must be explicit")
    if detrend not in {"none", "constant"}:
        raise ValueError("detrend must be 'none' or 'constant'")

    frequency_hz, density = scipy_signal.periodogram(
        values,
        fs=1.0 / float(sample_interval_seconds),
        window=PERIODOGRAM_WINDOW,
        detrend=False if detrend == "none" else "constant",
        return_onesided=True,
        scaling=PERIODOGRAM_SCALING,
    )
    return PeriodogramReport(
        frequency_hz=frequency_hz,
        power_spectral_density=density,
        sample_interval_seconds=float(sample_interval_seconds),
        signal_unit=signal_unit.strip(),
        detrend=detrend,
    )
