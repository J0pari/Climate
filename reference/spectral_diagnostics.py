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
COHERENCE_BACKEND = "scipy.signal.coherence"
COHERENCE_WINDOW = "hann"
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

@dataclass(frozen=True)
class CoherenceReport:
    frequency_hz: np.ndarray
    magnitude_squared_coherence: np.ndarray
    sample_interval_seconds: float
    x_unit: str
    y_unit: str
    segment_length: int
    overlap_samples: int
    detrend: str
    backend: str = COHERENCE_BACKEND
    window: str = COHERENCE_WINDOW
    missingness_policy: str = MISSINGNESS_POLICY
    frequency_unit: str = "Hz"
    coherence_unit: str = "1"

    def __post_init__(self) -> None:
        frequency = np.asarray(self.frequency_hz, dtype=np.float64)
        coherence = np.asarray(self.magnitude_squared_coherence, dtype=np.float64)
        if frequency.ndim != 1 or coherence.ndim != 1 or frequency.shape != coherence.shape:
            raise ValueError("frequency and coherence must be matching vectors")
        if frequency.size < 2:
            raise ValueError("coherence report requires at least two frequency bins")
        if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(coherence)):
            raise ValueError("coherence report must be finite")
        if np.any(frequency < 0.0) or np.any(np.diff(frequency) <= 0.0):
            raise ValueError("coherence frequencies must be strictly increasing and non-negative")
        roundoff = 64.0 * np.finfo(np.float64).eps
        if np.any(coherence < -roundoff) or np.any(coherence > 1.0 + roundoff):
            raise ValueError("magnitude-squared coherence must lie in [0,1] up to roundoff")
        if not np.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0.0:
            raise ValueError("sample interval must be finite and positive")
        if not self.x_unit.strip() or not self.y_unit.strip():
            raise ValueError("both signal units must be explicit")
        if self.segment_length < 4 or self.overlap_samples < 0 or self.overlap_samples >= self.segment_length:
            raise ValueError("invalid Welch segment or overlap policy")
        if self.detrend not in {"none", "constant"}:
            raise ValueError("unsupported detrend policy")
        frequency = np.array(frequency, copy=True)
        coherence = np.clip(np.array(coherence, copy=True), 0.0, 1.0)
        frequency.setflags(write=False)
        coherence.setflags(write=False)
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "magnitude_squared_coherence", coherence)


def magnitude_squared_coherence(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    *,
    sample_interval_seconds: float,
    x_unit: str,
    y_unit: str,
    segment_length: int,
    overlap_samples: int,
    detrend: str = "constant",
) -> CoherenceReport:
    """Compute Welch magnitude-squared coherence through SciPy without inference claims."""
    x = np.asarray(x_samples, dtype=np.float64)
    y = np.asarray(y_samples, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 4:
        raise ValueError("coherence requires matching one-dimensional signals with at least four samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coherence rejects non-finite or missing samples")
    if not np.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0.0:
        raise ValueError("sample interval must be finite and positive")
    if not isinstance(x_unit, str) or not x_unit.strip() or not isinstance(y_unit, str) or not y_unit.strip():
        raise ValueError("both signal units must be explicit")
    if not isinstance(segment_length, int) or segment_length < 4 or segment_length > x.size:
        raise ValueError("segment length must be an integer in [4, sample_count]")
    if not isinstance(overlap_samples, int) or overlap_samples < 0 or overlap_samples >= segment_length:
        raise ValueError("overlap samples must be an integer in [0, segment_length)")
    if detrend not in {"none", "constant"}:
        raise ValueError("detrend must be 'none' or 'constant'")

    frequency_hz, coherence = scipy_signal.coherence(
        x,
        y,
        fs=1.0 / float(sample_interval_seconds),
        window=COHERENCE_WINDOW,
        nperseg=segment_length,
        noverlap=overlap_samples,
        detrend=False if detrend == "none" else "constant",
    )
    return CoherenceReport(
        frequency_hz=frequency_hz,
        magnitude_squared_coherence=coherence,
        sample_interval_seconds=float(sample_interval_seconds),
        x_unit=x_unit.strip(),
        y_unit=y_unit.strip(),
        segment_length=segment_length,
        overlap_samples=overlap_samples,
        detrend=detrend,
    )

