from __future__ import annotations

import unittest

import numpy as np
from scipy import signal as scipy_signal

from reference.spectral_diagnostics import (
    COHERENCE_BACKEND,
    MISSINGNESS_POLICY,
    PERIODOGRAM_BACKEND,
    magnitude_squared_coherence,
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

    def test_broadband_linear_transform_has_unit_coherence(self) -> None:
        rng = np.random.default_rng(20260923)
        x = rng.normal(size=1024)
        y = 3.5 * x
        report = magnitude_squared_coherence(
            x,
            y,
            sample_interval_seconds=3600.0,
            x_unit="K",
            y_unit="W m-2",
            segment_length=256,
            overlap_samples=128,
            detrend="constant",
        )
        self.assertEqual(report.backend, COHERENCE_BACKEND)
        self.assertEqual(report.coherence_unit, "1")
        self.assertEqual(report.x_unit, "K")
        self.assertEqual(report.y_unit, "W m-2")
        np.testing.assert_allclose(
            report.magnitude_squared_coherence,
            1.0,
            rtol=0.0,
            atol=128.0 * np.finfo(np.float64).eps,
        )

    def test_phase_locked_sine_peaks_at_declared_frequency(self) -> None:
        sample_count = 1024
        segment_length = 256
        sample_interval = 1.0
        bin_index = 16
        frequency_hz = bin_index / (segment_length * sample_interval)
        time = np.arange(sample_count, dtype=np.float64) * sample_interval
        x = np.sin(2.0 * np.pi * frequency_hz * time)
        y = 2.0 * np.sin(2.0 * np.pi * frequency_hz * time + 0.7)
        report = magnitude_squared_coherence(
            x,
            y,
            sample_interval_seconds=sample_interval,
            x_unit="K",
            y_unit="K",
            segment_length=segment_length,
            overlap_samples=128,
            detrend="none",
        )
        target = int(np.argmin(np.abs(report.frequency_hz - frequency_hz)))
        self.assertEqual(report.frequency_hz[target], frequency_hz)
        self.assertGreater(report.magnitude_squared_coherence[target], 1.0 - 1.0e-12)

    def test_coherence_missingness_and_welch_policy_fail_closed(self) -> None:
        x = np.arange(16, dtype=np.float64)
        y = x.copy()
        y[3] = np.nan
        with self.assertRaisesRegex(ValueError, "rejects non-finite"):
            magnitude_squared_coherence(
                x,
                y,
                sample_interval_seconds=1.0,
                x_unit="K",
                y_unit="K",
                segment_length=8,
                overlap_samples=4,
            )
        with self.assertRaisesRegex(ValueError, "segment length"):
            magnitude_squared_coherence(
                x,
                x,
                sample_interval_seconds=1.0,
                x_unit="K",
                y_unit="K",
                segment_length=32,
                overlap_samples=4,
            )
        with self.assertRaisesRegex(ValueError, "overlap samples"):
            magnitude_squared_coherence(
                x,
                x,
                sample_interval_seconds=1.0,
                x_unit="K",
                y_unit="K",
                segment_length=8,
                overlap_samples=8,
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
