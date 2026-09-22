from __future__ import annotations

import unittest

import numpy as np
from scipy import signal as scipy_signal

from reference.spectral_diagnostics import (
    MISSINGNESS_POLICY,
    PERIODOGRAM_BACKEND,
    one_sided_periodogram,
)


class SpectralDiagnosticsTests(unittest.TestCase):
    def test_bin_aligned_sine_recovers_frequency_and_mean_square_power(self) -> None:
        sample_interval = 3600.0
        sample_count = 256
        bin_index = 8
        frequency_hz = bin_index / (sample_count * sample_interval)
        time = np.arange(sample_count, dtype=np.float64) * sample_interval
        amplitude = 3.0
        values = amplitude * np.sin(2.0 * np.pi * frequency_hz * time)

        report = one_sided_periodogram(
            values,
            sample_interval_seconds=sample_interval,
            signal_unit="K",
            detrend="none",
        )
        peak = int(np.argmax(report.power_spectral_density[1:]) + 1)
        self.assertEqual(report.frequency_hz[peak], frequency_hz)
        self.assertAlmostEqual(report.integrated_power(), amplitude**2 / 2.0, places=12)
        self.assertEqual(report.power_spectral_density_unit, "(K)^2/Hz")
        self.assertEqual(report.backend, PERIODOGRAM_BACKEND)
        self.assertEqual(report.missingness_policy, MISSINGNESS_POLICY)

    def test_constant_detrend_integrated_power_matches_sample_variance(self) -> None:
        rng = np.random.default_rng(20260922)
        values = 280.0 + rng.normal(0.0, 2.0, size=512)
        report = one_sided_periodogram(
            values,
            sample_interval_seconds=86400.0,
            signal_unit="K",
            detrend="constant",
        )
        expected = float(np.mean((values - np.mean(values)) ** 2))
        self.assertAlmostEqual(report.integrated_power(), expected, places=12)
        numerical_zero = (
            16.0
            * np.finfo(np.float64).eps
            * float(np.max(report.power_spectral_density))
        )
        self.assertLessEqual(abs(report.power_spectral_density[0]), numerical_zero)

    def test_matches_scipy_periodogram_without_project_local_fft(self) -> None:
        values = np.asarray([1.0, -1.0, 2.0, -2.0, 0.5, -0.5], dtype=np.float64)
        report = one_sided_periodogram(
            values,
            sample_interval_seconds=2.0,
            signal_unit="m s-1",
            detrend="none",
        )
        expected_frequency, expected_density = scipy_signal.periodogram(
            values,
            fs=0.5,
            window="boxcar",
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        np.testing.assert_array_equal(report.frequency_hz, expected_frequency)
        np.testing.assert_array_equal(report.power_spectral_density, expected_density)

    def test_nonfinite_samples_fail_closed_instead_of_interpolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "rejects non-finite"):
            one_sided_periodogram(
                np.asarray([1.0, np.nan, 2.0]),
                sample_interval_seconds=1.0,
                signal_unit="K",
            )

    def test_sampling_units_and_detrend_policy_are_explicit(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "sample interval"):
            one_sided_periodogram(values, sample_interval_seconds=0.0, signal_unit="K")
        with self.assertRaisesRegex(ValueError, "signal unit"):
            one_sided_periodogram(values, sample_interval_seconds=1.0, signal_unit="")
        with self.assertRaisesRegex(ValueError, "detrend"):
            one_sided_periodogram(
                values,
                sample_interval_seconds=1.0,
                signal_unit="K",
                detrend="linear",
            )


if __name__ == "__main__":
    unittest.main()
