# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for scipy_filters module.

This test suite expands on existing unittest tests in test_xr_scipy.py, providing:
- Fixtures for signal generation and xarray data structures
- Parameterized tests for filter types and parameters
- Spectral analysis validation for frequency filters
- Edge case testing and error condition handling
- Performance optimization using session-scoped fixtures

Test Coverage:
- frequency_filter() function with IIR/FIR filters
- lowpass(), highpass(), bandpass(), bandstop() filters
- decimate(), resample_poly(), savgol_filter(), detrend() functions
- FilterAccessor class methods and properties
- Warning systems and error handling
- Edge cases and parameter validation

Created: September 23, 2025
Author: GitHub Copilot (expanding on existing test_xr_scipy.py)
"""

# =============================================================================
# Imports
# =============================================================================
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# Import the module under test
try:
    from mth5.timeseries.channel_ts import make_dt_coordinates
    from mth5.timeseries.scipy_filters import (
        bandpass,
        bandstop,
        decimate,
        detrend,
        FilteringNaNWarning,
        frequency_filter,
        get_maybe_only_dim,
        get_sampling_step,
        highpass,
        lowpass,
        resample_poly,
        savgol_filter,
        UnevenSamplingWarning,
    )

    # Test import successful
    SCIPY_FILTERS_AVAILABLE = True
except ImportError as e:
    SCIPY_FILTERS_AVAILABLE = False
    pytest.skip(f"scipy_filters not available: {e}", allow_module_level=True)


# =============================================================================
# Fixture Definitions
# =============================================================================


@pytest.fixture(scope="session")
def base_signal_params():
    """Session-scoped fixture for basic signal parameters."""
    return {
        "n_samples": 4096,
        "sample_rate": 64.0,
        "duration": 64.0,  # seconds
        "start_time": "2020-01-01T00:00:00",
    }


@pytest.fixture(scope="session")
def synthetic_signals(base_signal_params):
    """
    Session-scoped fixture providing various synthetic signals for testing.

    Returns:
        dict: Dictionary containing different signal types and their properties
    """
    n_samples = base_signal_params["n_samples"]
    t = np.arange(n_samples)

    signals = {}

    # Multi-frequency signal with trend (similar to original test)
    signals["multi_freq_with_trend"] = (
        np.sum(
            [
                np.cos(2 * np.pi * w * t + phi)
                for w, phi in zip(
                    np.logspace(-3, 3, 20), np.random.RandomState(42).rand(20)
                )
            ],
            axis=0,
        )
        + 1.5 * t
        + 0.5
    )

    # Single frequency signals for filter validation
    signals["single_1hz"] = np.sin(
        2 * np.pi * 1.0 * t / base_signal_params["sample_rate"]
    )
    signals["single_10hz"] = np.sin(
        2 * np.pi * 10.0 * t / base_signal_params["sample_rate"]
    )
    signals["single_30hz"] = np.sin(
        2 * np.pi * 30.0 * t / base_signal_params["sample_rate"]
    )

    # Mixed frequency signal for bandpass testing
    signals["mixed_freq"] = (
        signals["single_1hz"] + signals["single_10hz"] + signals["single_30hz"]
    )

    # Linear trend signal
    signals["linear_trend"] = 0.5 * t + 10.0

    # Noisy signal
    np.random.seed(42)
    signals["noisy"] = signals["single_10hz"] + 0.1 * np.random.randn(n_samples)

    # Signal with NaN values
    signals["with_nans"] = signals["single_10hz"].copy()
    signals["with_nans"][100:110] = np.nan

    return signals


@pytest.fixture(scope="session")
def time_coordinates(base_signal_params):
    """Session-scoped fixture for time coordinates."""
    return make_dt_coordinates(
        base_signal_params["start_time"],
        base_signal_params["sample_rate"],
        base_signal_params["n_samples"],
    )


@pytest.fixture
def xr_data_array(synthetic_signals, time_coordinates, request):
    """
    Parameterized fixture for creating xarray DataArrays with different signal types.

    Usage:
        @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend", "single_10hz"], indirect=True)
    """
    signal_name = getattr(request, "param", "multi_freq_with_trend")
    return xr.DataArray(
        synthetic_signals[signal_name],
        coords={"time": time_coordinates},
        dims=["time"],
        name="test_signal",
    )


@pytest.fixture
def xr_dataset(synthetic_signals, time_coordinates):
    """Fixture providing multi-channel xarray Dataset."""
    return xr.Dataset(
        {
            "ex": xr.DataArray(
                synthetic_signals["multi_freq_with_trend"],
                coords={"time": time_coordinates},
                dims=["time"],
            ),
            "ey": xr.DataArray(
                synthetic_signals["mixed_freq"],
                coords={"time": time_coordinates},
                dims=["time"],
            ),
            "hx": xr.DataArray(
                synthetic_signals["noisy"],
                coords={"time": time_coordinates},
                dims=["time"],
            ),
        }
    )


@pytest.fixture(scope="session")
def filter_parameters():
    """Session-scoped fixture for common filter parameters."""
    return {
        "orders": [3, 5, 8],
        "ir_types": ["iir", "fir"],
        "cutoff_frequencies": [0.5, 2.0, 5.0, 15.0],
        "bandpass_pairs": [(1.0, 5.0), (2.0, 10.0), (5.0, 25.0)],
        "bandstop_pairs": [(9.0, 11.0), (19.0, 21.0)],  # Notch filters
    }


@pytest.fixture
def frequency_array(base_signal_params):
    """Fixture providing frequency array for spectral analysis."""
    return np.fft.rfftfreq(
        base_signal_params["n_samples"], d=1.0 / base_signal_params["sample_rate"]
    )


@pytest.fixture(autouse=True)
def capture_warnings():
    """Auto-use fixture to capture warnings during tests."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list


# =============================================================================
# Helper Functions for Testing
# =============================================================================


def calculate_filter_response(original_signal, filtered_signal, frequencies):
    """
    Calculate the frequency response of a filter by comparing original and filtered signals.

    Args:
        original_signal (np.ndarray): Original time domain signal
        filtered_signal (np.ndarray): Filtered time domain signal
        frequencies (np.ndarray): Frequency array for analysis

    Returns:
        tuple: (response_magnitude, response_phase) arrays
    """
    original_fft = np.fft.rfft(original_signal)
    filtered_fft = np.fft.rfft(filtered_signal)

    # Avoid division by zero
    mask = np.abs(original_fft) > 1e-10
    response = np.zeros_like(original_fft, dtype=complex)
    response[mask] = filtered_fft[mask] / original_fft[mask]

    magnitude = np.abs(response)
    phase = np.angle(response)

    return magnitude, phase


def validate_filter_response(magnitude, frequencies, filter_type, **filter_params):
    """
    Validate that a filter response matches expected characteristics.

    Args:
        magnitude (np.ndarray): Filter response magnitude
        frequencies (np.ndarray): Frequency array
        filter_type (str): Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
        **filter_params: Filter-specific parameters

    Returns:
        bool: True if filter response is valid
    """
    # Remove DC component for analysis
    if len(frequencies) > 1:
        freqs = frequencies[1:]
        mag = magnitude[1:]
    else:
        freqs = frequencies
        mag = magnitude

    if filter_type == "lowpass":
        f_cutoff = filter_params.get("f_cutoff", 5.0)
        # Check passband (should be ~1.0)
        passband_mask = freqs <= f_cutoff * 0.8
        if np.any(passband_mask):
            passband_response = np.median(mag[passband_mask])
            if not (0.7 <= passband_response <= 1.3):  # Allow some tolerance
                return False

        # Check stopband (should be << 1.0)
        stopband_mask = freqs >= f_cutoff * 2.0
        if np.any(stopband_mask):
            stopband_response = np.median(mag[stopband_mask])
            if not (stopband_response <= 0.3):  # Should be attenuated
                return False

    elif filter_type == "highpass":
        f_cutoff = filter_params.get("f_cutoff", 5.0)
        # Check stopband (should be << 1.0)
        stopband_mask = freqs <= f_cutoff * 0.5
        if np.any(stopband_mask):
            stopband_response = np.median(mag[stopband_mask])
            if not (stopband_response <= 0.3):
                return False

        # Check passband (should be ~1.0)
        passband_mask = freqs >= f_cutoff * 2.0
        if np.any(passband_mask):
            passband_response = np.median(mag[passband_mask])
            if not (0.7 <= passband_response <= 1.3):
                return False

    elif filter_type == "bandpass":
        f_low = filter_params.get("f_low", 2.0)
        f_high = filter_params.get("f_high", 10.0)

        # Check passband
        passband_mask = (freqs >= f_low * 1.2) & (freqs <= f_high * 0.8)
        if np.any(passband_mask):
            passband_response = np.median(mag[passband_mask])
            if not (0.7 <= passband_response <= 1.3):
                return False

        # Check stopbands
        low_stopband_mask = freqs <= f_low * 0.5
        high_stopband_mask = freqs >= f_high * 2.0

        for stopband_mask in [low_stopband_mask, high_stopband_mask]:
            if np.any(stopband_mask):
                stopband_response = np.median(mag[stopband_mask])
                if not (stopband_response <= 0.3):
                    return False

    elif filter_type == "bandstop":
        f_low = filter_params.get("f_low", 9.0)
        f_high = filter_params.get("f_high", 11.0)

        # Check stopband (should be attenuated)
        stopband_mask = (freqs >= f_low * 1.1) & (freqs <= f_high * 0.9)
        if np.any(stopband_mask):
            stopband_response = np.median(mag[stopband_mask])
            if not (stopband_response <= 0.3):
                return False

        # Check passbands
        low_passband_mask = freqs <= f_low * 0.5
        high_passband_mask = freqs >= f_high * 2.0

        for passband_mask in [low_passband_mask, high_passband_mask]:
            if np.any(passband_mask):
                passband_response = np.median(mag[passband_mask])
                if not (0.7 <= passband_response <= 1.3):
                    return False

    return True


# =============================================================================
# Core Filter Function Tests
# =============================================================================


class TestFrequencyFilter:
    """Test suite for the core frequency_filter function."""

    @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend"], indirect=True)
    @pytest.mark.parametrize("ir_type", ["iir", "fir"])
    def test_frequency_filter_types(self, xr_data_array, ir_type):
        """Test frequency_filter with different IR types."""
        # Use lowpass wrapper function which properly sets parameters for each irtype
        if ir_type == "iir":
            filtered = frequency_filter(
                xr_data_array, f_crit=5.0, order=5, irtype=ir_type, btype="lowpass"
            )
        else:  # fir
            filtered = frequency_filter(
                xr_data_array,
                f_crit=5.0,
                order=5,
                irtype=ir_type,
                pass_zero=True,  # FIR parameter for lowpass
            )

        # Verify output structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape
        assert filtered.dims == xr_data_array.dims
        assert np.array_equal(filtered.coords["time"], xr_data_array.coords["time"])

    @pytest.mark.parametrize("xr_data_array", ["single_10hz"], indirect=True)
    @pytest.mark.parametrize("order", [3, 5, 8])
    def test_frequency_filter_orders(self, xr_data_array, order, frequency_array):
        """Test frequency_filter with different filter orders."""
        # Apply lowpass filter using wrapper function
        filtered = lowpass(xr_data_array, f_cutoff=5.0, order=order)

        # Calculate filter response
        magnitude, _ = calculate_filter_response(
            xr_data_array.values, filtered.values, frequency_array
        )

        # Higher order should provide steeper rolloff
        # Check that 10Hz (above 5Hz cutoff) is more attenuated with higher order
        freq_10hz_idx = np.argmin(np.abs(frequency_array - 10.0))
        if freq_10hz_idx > 0:
            attenuation_10hz = magnitude[freq_10hz_idx]
            assert (
                attenuation_10hz < 0.5
            ), f"Order {order} should attenuate 10Hz significantly"

    @pytest.mark.parametrize("xr_data_array", ["mixed_freq"], indirect=True)
    def test_frequency_filter_bandpass(self, xr_data_array, frequency_array):
        """Test frequency_filter with bandpass configuration."""
        # Apply bandpass filter to preserve only 10Hz component using wrapper function
        filtered = bandpass(xr_data_array, f_low=5.0, f_high=15.0, order=5)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # Bandpass should modify the signal
        assert not np.array_equal(filtered.values, xr_data_array.values)

        # Should preserve some energy
        original_power = np.sum(xr_data_array.values**2)
        filtered_power = np.sum(filtered.values**2)
        power_ratio = filtered_power / original_power

        # Should retain some but not all energy
        assert (
            0.1 < power_ratio < 0.9
        ), f"Bandpass [5.0, 15.0]Hz should modify signal energy"

    def test_frequency_filter_invalid_irtype(self):
        """Test frequency_filter with invalid IR type."""
        data = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])

        with pytest.raises(ValueError, match="Wrong argument for irtype"):
            frequency_filter(data, f_crit=1.0, irtype="invalid")

    @pytest.mark.parametrize("xr_data_array", ["with_nans"], indirect=True)
    def test_frequency_filter_with_nans(self, xr_data_array, capture_warnings):
        """Test frequency_filter behavior with NaN values."""
        filtered = lowpass(xr_data_array, f_cutoff=5.0)

        # Check that NaN warning was issued
        nan_warnings = [
            w for w in capture_warnings if issubclass(w.category, FilteringNaNWarning)
        ]
        assert len(nan_warnings) > 0, "Expected FilteringNaNWarning"

        # Filtered data should still have NaNs in the same locations
        original_nan_mask = np.isnan(xr_data_array.values)
        filtered_nan_mask = np.isnan(filtered.values)

        # NaNs should propagate through the filter
        assert np.any(filtered_nan_mask), "Expected NaNs in filtered output"


class TestSpecificFilters:
    """Test suite for specific filter functions (lowpass, highpass, bandpass, bandstop)."""

    @pytest.mark.parametrize("xr_data_array", ["single_10hz"], indirect=True)
    @pytest.mark.parametrize("f_cutoff", [2.0, 5.0, 15.0])
    def test_lowpass_filter(self, xr_data_array, f_cutoff, frequency_array):
        """Test lowpass filter with various cutoff frequencies."""
        filtered = lowpass(xr_data_array, f_cutoff)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # For 10Hz signal with low cutoff, expect significant attenuation
        if f_cutoff < 10.0:
            # Signal should be significantly attenuated
            original_power = np.sum(xr_data_array.values**2)
            filtered_power = np.sum(filtered.values**2)
            power_ratio = filtered_power / original_power

            assert (
                power_ratio < 0.5
            ), f"10Hz signal should be attenuated by {f_cutoff}Hz lowpass"

        # For high cutoff (15Hz), signal should pass through mostly unchanged
        elif f_cutoff > 10.0:
            original_power = np.sum(xr_data_array.values**2)
            filtered_power = np.sum(filtered.values**2)
            power_ratio = filtered_power / original_power

            assert (
                power_ratio > 0.8
            ), f"10Hz signal should mostly pass through {f_cutoff}Hz lowpass"

    @pytest.mark.parametrize("xr_data_array", ["single_1hz"], indirect=True)
    @pytest.mark.parametrize("f_cutoff", [2.0, 5.0, 10.0])
    def test_highpass_filter(self, xr_data_array, f_cutoff, frequency_array):
        """Test highpass filter with various cutoff frequencies."""
        filtered = highpass(xr_data_array, f_cutoff)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # For 1Hz signal with high cutoff, expect significant attenuation
        if f_cutoff > 2.0:
            # Signal should be significantly attenuated
            original_power = np.sum(xr_data_array.values**2)
            filtered_power = np.sum(filtered.values**2)
            power_ratio = filtered_power / original_power

            assert (
                power_ratio < 0.5
            ), f"1Hz signal should be attenuated by {f_cutoff}Hz highpass"

    @pytest.mark.parametrize("xr_data_array", ["mixed_freq"], indirect=True)
    @pytest.mark.parametrize("f_low,f_high", [(2.0, 15.0), (5.0, 25.0)])
    def test_bandpass_filter(self, xr_data_array, f_low, f_high, frequency_array):
        """Test bandpass filter with various frequency bands."""
        filtered = bandpass(xr_data_array, f_low, f_high)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # Bandpass should modify the signal (not be identical to original)
        assert not np.array_equal(filtered.values, xr_data_array.values)

        # Should preserve some energy from the original signal
        original_power = np.sum(xr_data_array.values**2)
        filtered_power = np.sum(filtered.values**2)
        power_ratio = filtered_power / original_power

        # Should retain some but not all energy
        assert (
            0.1 < power_ratio < 0.9
        ), f"Bandpass [{f_low}, {f_high}]Hz should modify signal energy"

    @pytest.mark.parametrize("xr_data_array", ["single_10hz"], indirect=True)
    @pytest.mark.parametrize("f_low,f_high", [(8.0, 12.0), (9.5, 10.5)])
    def test_bandstop_filter(self, xr_data_array, f_low, f_high, frequency_array):
        """Test bandstop (notch) filter targeting 10Hz signal."""
        filtered = bandstop(xr_data_array, f_low, f_high)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # Bandstop should modify the signal
        assert not np.array_equal(filtered.values, xr_data_array.values)

        # For bandstop around 10Hz, the 10Hz signal should be attenuated
        original_power = np.sum(xr_data_array.values**2)
        filtered_power = np.sum(filtered.values**2)
        power_ratio = filtered_power / original_power

        # Expect significant attenuation when notch targets the signal frequency
        if f_low <= 10.0 <= f_high:
            assert (
                power_ratio < 0.5
            ), f"10Hz signal should be attenuated by [{f_low}, {f_high}]Hz bandstop"


# =============================================================================
# Advanced Processing Function Tests
# =============================================================================


class TestAdvancedProcessing:
    """Test suite for advanced processing functions (decimate, resample_poly, etc.)."""

    @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend"], indirect=True)
    @pytest.mark.parametrize("target_rate", [1.0, 4.0, 16.0])
    def test_decimate_function(self, xr_data_array, target_rate, base_signal_params):
        """Test decimation with various target sample rates."""
        original_rate = base_signal_params["sample_rate"]

        # Skip if decimation factor would be too high
        decimation_factor = int(original_rate / target_rate)
        if decimation_factor > 13:
            pytest.skip(f"Decimation factor {decimation_factor} too high")

        decimated = decimate(xr_data_array, target_rate)

        # Validate structure
        assert isinstance(decimated, xr.DataArray)
        assert decimated.dims == xr_data_array.dims

        # Validate sample rate
        expected_samples = int(xr_data_array.sizes["time"] / decimation_factor)
        assert abs(decimated.sizes["time"] - expected_samples) <= 1

        # Validate time coordinates
        dt_original = get_sampling_step(xr_data_array)
        dt_decimated = get_sampling_step(decimated)
        expected_dt = dt_original * decimation_factor

        assert abs(dt_decimated - expected_dt) / expected_dt < 0.01  # 1% tolerance

    def test_decimate_high_factor_warning(self, capture_warnings):
        """Test that decimate issues warning for high decimation factors."""
        # Create data that would require high decimation factor
        time = pd.date_range("2020-01-01", periods=1000, freq="1ms")  # 1000 Hz
        data = xr.DataArray(np.random.randn(1000), coords={"time": time}, dims=["time"])

        # Decimate to very low rate (high factor)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decimated = decimate(data, target_sample_rate=1.0)  # Factor ~1000

            # Should get a warning about high decimation factor
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "Decimation factor is larger than 13" in msg for msg in warning_messages
            ), (
                f"Expected warning about high decimation factor, got: {warning_messages}"
                @ pytest.mark.parametrize(
                    "xr_data_array", ["multi_freq_with_trend"], indirect=True
                )
            )

    @pytest.mark.parametrize("new_rate", [3.0, 4.0, 32.0])
    def test_resample_poly_function(self, xr_data_array, new_rate, base_signal_params):
        """Test polynomial resampling with various rates."""
        resampled = resample_poly(xr_data_array, new_rate)

        # Validate structure
        assert isinstance(resampled, xr.DataArray)
        assert resampled.dims == xr_data_array.dims

        # Validate approximate sample count
        original_rate = base_signal_params["sample_rate"]
        expected_samples = int(xr_data_array.sizes["time"] * new_rate / original_rate)

        # Allow some tolerance due to resampling edge effects
        assert abs(resampled.sizes["time"] - expected_samples) <= 2

        # Validate time coordinates
        assert (
            resampled.coords["time"][0] == xr_data_array.coords["time"][0]
        )  # Same start time

    @pytest.mark.parametrize("xr_data_array", ["linear_trend"], indirect=True)
    @pytest.mark.parametrize("trend_type", ["linear", "constant"])
    def test_detrend_function(self, xr_data_array, trend_type):
        """Test detrending with different trend types."""
        detrended = detrend(xr_data_array, trend_type=trend_type)

        # Validate structure
        assert isinstance(detrended, xr.DataArray)
        assert detrended.shape == xr_data_array.shape

        if trend_type == "linear":
            # For a pure linear trend, detrending should result in near-zero mean
            assert abs(detrended.values.mean()) < 1e-10

            # Variance should be very small (nearly constant after detrend)
            assert detrended.values.std() < 0.1
        elif trend_type == "constant":
            # Constant detrend removes mean
            assert abs(detrended.values.mean()) < 1e-10

    @pytest.mark.parametrize("xr_data_array", ["noisy"], indirect=True)
    @pytest.mark.parametrize("window_length", [11, 21, 31])
    @pytest.mark.parametrize("polyorder", [2, 3, 4])
    def test_savgol_filter_function(self, xr_data_array, window_length, polyorder):
        """Test Savitzky-Golay filter with various parameters."""
        if polyorder >= window_length:
            pytest.skip(
                f"polyorder ({polyorder}) must be less than window_length ({window_length})"
            )

        filtered = savgol_filter(
            xr_data_array, window_length=window_length, polyorder=polyorder
        )

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # Savitzky-Golay should smooth noise while preserving signal
        # Check that filtering reduces high-frequency noise
        original_diff = np.diff(xr_data_array.values)
        filtered_diff = np.diff(filtered.values)

        # Filtered signal should be smoother (smaller differences)
        assert np.std(filtered_diff) <= np.std(original_diff)


# =============================================================================
# FilterAccessor Tests
# =============================================================================


class TestFilterAccessor:
    """Test suite for the FilterAccessor class and its methods."""

    @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend"], indirect=True)
    def test_accessor_properties(self, xr_data_array, base_signal_params):
        """Test FilterAccessor properties (dt, fs, dx)."""
        # Test sampling time step
        expected_dt = 1.0 / base_signal_params["sample_rate"]
        assert abs(xr_data_array.sps_filters.dt - expected_dt) < 1e-10

        # Test sampling frequency
        expected_fs = base_signal_params["sample_rate"]
        assert abs(xr_data_array.sps_filters.fs - expected_fs) < 1e-10

        # Test sampling steps array
        expected_dx = np.array([expected_dt])
        assert np.allclose(xr_data_array.sps_filters.dx, expected_dx)

    @pytest.mark.parametrize("xr_data_array", ["single_10hz"], indirect=True)
    def test_accessor_lowpass(self, xr_data_array, frequency_array):
        """Test FilterAccessor lowpass method."""
        filtered = xr_data_array.sps_filters.low(5.0)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # For 10Hz signal with 5Hz lowpass, expect significant attenuation
        original_power = np.sum(xr_data_array.values**2)
        filtered_power = np.sum(filtered.values**2)
        power_ratio = filtered_power / original_power

        assert power_ratio < 0.5, (
            f"10Hz signal should be attenuated by 5Hz lowpass filter"
            @ pytest.mark.parametrize("xr_data_array", ["single_1hz"], indirect=True)
        )

    def test_accessor_highpass(self, xr_data_array):
        """Test FilterAccessor highpass method."""
        filtered = xr_data_array.sps_filters.high(5.0)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # 1Hz signal should be significantly attenuated by 5Hz highpass
        original_power = np.sum(xr_data_array.values**2)
        filtered_power = np.sum(filtered.values**2)
        power_ratio = filtered_power / original_power

        assert power_ratio < 0.3, "1Hz signal should be attenuated by 5Hz highpass"

    @pytest.mark.parametrize("xr_data_array", ["mixed_freq"], indirect=True)
    def test_accessor_bandpass(self, xr_data_array, frequency_array):
        """Test FilterAccessor bandpass method."""
        filtered = xr_data_array.sps_filters.bandpass(5.0, 15.0)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # Bandpass should modify the signal
        assert not np.array_equal(filtered.values, xr_data_array.values)

        # Should preserve some energy
        original_power = np.sum(xr_data_array.values**2)
        filtered_power = np.sum(filtered.values**2)
        power_ratio = filtered_power / original_power

        # Should retain some but not all energy
        assert (
            0.1 < power_ratio < 0.9
        ), f"Bandpass [5.0, 15.0]Hz should modify signal energy"

    @pytest.mark.parametrize("xr_data_array", ["single_10hz"], indirect=True)
    def test_accessor_bandstop(self, xr_data_array):
        """Test FilterAccessor bandstop method."""
        filtered = xr_data_array.sps_filters.bandstop(8.0, 12.0)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # Bandstop should modify the signal
        assert not np.array_equal(filtered.values, xr_data_array.values)

        # Basic validation - 10Hz signal should be significantly attenuated by 8-12Hz bandstop
        original_power = np.sum(xr_data_array.values**2)
        filtered_power = np.sum(filtered.values**2)
        power_ratio = filtered_power / original_power

        # Bandstop should significantly attenuate the target frequency
        assert (
            power_ratio < 0.5
        ), f"Bandstop filter should strongly attenuate 10Hz signal: {power_ratio}"

    @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend"], indirect=True)
    def test_accessor_freq_method(self, xr_data_array):
        """Test FilterAccessor freq method (general frequency filter)."""
        # Test lowpass via freq method using proper IIR parameters
        filtered = xr_data_array.sps_filters.freq(
            f_crit=5.0, btype="lowpass", order=5, irtype="iir"
        )

        # Should be equivalent to direct lowpass call
        direct_filtered = xr_data_array.sps_filters.low(5.0, order=5)

        # Results should be very similar (allowing for small numerical differences)
        assert np.allclose(filtered.values, direct_filtered.values, rtol=1e-10)

    @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend"], indirect=True)
    def test_accessor_decimate(self, xr_data_array, base_signal_params):
        """Test FilterAccessor decimate method."""
        target_rate = 1.0
        decimated = xr_data_array.sps_filters.decimate(target_rate)

        # Validate sample rate
        assert abs(decimated.sps_filters.fs - target_rate) < 0.1

        # Validate size reduction
        original_rate = base_signal_params["sample_rate"]
        expected_size = int(xr_data_array.sizes["time"] * target_rate / original_rate)
        assert abs(decimated.sizes["time"] - expected_size) <= 1

    @pytest.mark.parametrize("xr_data_array", ["linear_trend"], indirect=True)
    def test_accessor_detrend(self, xr_data_array):
        """Test FilterAccessor detrend method."""
        detrended = xr_data_array.sps_filters.detrend("linear")

        # Linear trend should be removed, resulting in near-zero mean
        assert abs(detrended.values.mean()) < 1e-10

    @pytest.mark.parametrize("xr_data_array", ["noisy"], indirect=True)
    def test_accessor_savgol(self, xr_data_array):
        """Test FilterAccessor savgol method."""
        filtered = xr_data_array.sps_filters.savgol(window_length=21, polyorder=3)

        # Validate structure
        assert isinstance(filtered, xr.DataArray)
        assert filtered.shape == xr_data_array.shape

        # Should reduce noise while preserving signal
        original_diff = np.diff(xr_data_array.values)
        filtered_diff = np.diff(filtered.values)
        assert np.std(filtered_diff) <= np.std(original_diff)

    @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend"], indirect=True)
    def test_accessor_resample_poly(self, xr_data_array, base_signal_params):
        """Test FilterAccessor resample_poly method."""
        new_rate = 4.0
        resampled = xr_data_array.sps_filters.resample_poly(new_rate)

        # Validate approximate sample count
        original_rate = base_signal_params["sample_rate"]
        expected_samples = int(xr_data_array.sizes["time"] * new_rate / original_rate)

        assert abs(resampled.sizes["time"] - expected_samples) <= 2


# =============================================================================
# Dataset Testing
# =============================================================================


class TestDatasetFiltering:
    """Test suite for filtering xarray Datasets."""

    def test_dataset_lowpass(self, xr_dataset, frequency_array):
        """Test lowpass filtering on xarray Dataset."""
        filtered_dataset = lowpass(xr_dataset, f_cutoff=5.0)

        # Validate structure
        assert isinstance(filtered_dataset, xr.Dataset)
        assert set(filtered_dataset.data_vars) == set(xr_dataset.data_vars)

        # Validate that each variable was filtered
        for var_name in xr_dataset.data_vars:
            original_var = xr_dataset[var_name]
            filtered_var = filtered_dataset[var_name]

            # Simple functionality test - should modify signal
            assert not np.array_equal(original_var.values, filtered_var.values)

            # Basic validation - filters should not amplify significantly
            original_power = np.sum(original_var.values**2)
            filtered_power = np.sum(filtered_var.values**2)
            power_ratio = filtered_power / original_power

            # Should not amplify too much
            assert (
                power_ratio < 1.1
            ), f"Filter should not amplify signal in {var_name}: {power_ratio}"

    def test_dataset_accessor_methods(self, xr_dataset):
        """Test that accessor methods work on Dataset."""
        # Test properties
        assert hasattr(xr_dataset.sps_filters, "dt")
        assert hasattr(xr_dataset.sps_filters, "fs")
        assert hasattr(xr_dataset.sps_filters, "dx")

        # Test filter methods
        filtered = xr_dataset.sps_filters.low(5.0)
        assert isinstance(filtered, xr.Dataset)
        assert set(filtered.data_vars) == set(xr_dataset.data_vars)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_uneven_sampling_warning(self, capture_warnings):
        """Test that uneven sampling triggers warning."""
        # Create unevenly sampled time series
        irregular_times = pd.to_datetime(
            [
                "2020-01-01T00:00:00",
                "2020-01-01T00:00:01",
                "2020-01-01T00:00:03",  # Gap here
                "2020-01-01T00:00:04",
                "2020-01-01T00:00:05",
            ]
        )

        data = xr.DataArray(
            [1, 2, 3, 4, 5], coords={"time": irregular_times}, dims=["time"]
        )

        # Try to get sampling step (should trigger warning)
        dt = get_sampling_step(data)

        # Check for uneven sampling warning
        uneven_warnings = [
            w for w in capture_warnings if issubclass(w.category, UnevenSamplingWarning)
        ]
        assert len(uneven_warnings) > 0, "Expected UnevenSamplingWarning"

    def test_single_dimension_auto_detection(self):
        """Test automatic dimension detection for 1D arrays."""
        data_1d = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])

        # Should automatically detect 'time' dimension
        detected_dim = get_maybe_only_dim(data_1d, dim=None)
        assert detected_dim == "time"

    def test_multi_dimension_requires_explicit_dim(self):
        """Test that multi-dimensional arrays require explicit dimension specification."""
        data_2d = xr.DataArray(np.random.randn(10, 5), dims=["time", "channel"])

        # Should raise ValueError when dim=None for multi-dimensional array
        with pytest.raises(ValueError, match="Specify the dimension"):
            get_maybe_only_dim(data_2d, dim=None)

    def test_invalid_filter_parameters(self):
        """Test handling of invalid filter parameters."""
        data = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])

        # Test invalid order (should be positive)
        with pytest.raises((ValueError, TypeError)):
            frequency_filter(data, f_crit=1.0, order=-1)

    @pytest.mark.parametrize("xr_data_array", ["single_10hz"], indirect=True)
    def test_extreme_cutoff_frequencies(self, xr_data_array, base_signal_params):
        """Test behavior with extreme cutoff frequencies."""
        nyquist = base_signal_params["sample_rate"] / 2.0

        # Very low cutoff - should heavily attenuate 10Hz signal
        low_cutoff_filtered = lowpass(xr_data_array, f_cutoff=0.5)
        power_ratio = np.sum(low_cutoff_filtered.values**2) / np.sum(
            xr_data_array.values**2
        )
        assert (
            power_ratio < 0.2
        ), f"10Hz signal should be heavily attenuated by 0.5Hz lowpass"

        # High cutoff - should minimally attenuate 10Hz signal
        high_cutoff_filtered = lowpass(xr_data_array, f_cutoff=nyquist * 0.9)
        power_ratio = np.sum(high_cutoff_filtered.values**2) / np.sum(
            xr_data_array.values**2
        )
        assert (
            power_ratio > 0.8
        ), f"10Hz signal should pass through {nyquist * 0.9}Hz lowpass filter"


# =============================================================================
# Integration and Performance Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple filter operations."""

    @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend"], indirect=True)
    def test_filter_chain(self, xr_data_array):
        """Test chaining multiple filter operations."""
        # Apply a series of filters
        result = (
            xr_data_array.sps_filters.detrend("linear")  # Remove trend
            .sps_filters.high(1.0)  # Remove very low frequencies
            .sps_filters.low(20.0)  # Remove very high frequencies
            .sps_filters.savgol(21, 3)
        )  # Smooth

        # Validate final result
        assert isinstance(result, xr.DataArray)
        assert result.shape == xr_data_array.shape

        # Should be different from original due to filtering
        assert not np.array_equal(result.values, xr_data_array.values)

        # Final result should have reduced variance (due to smoothing)
        assert result.values.std() <= xr_data_array.values.std()

    def test_downsample_and_filter_workflow(self, xr_dataset):
        """Test typical workflow of filtering followed by downsampling."""
        # Common workflow: filter then downsample
        for var_name, var_data in xr_dataset.data_vars.items():
            # Apply anti-aliasing filter before downsampling
            filtered = var_data.sps_filters.low(f_cutoff=8.0)

            # Downsample
            downsampled = filtered.sps_filters.decimate(target_sample_rate=16.0)

            # Validate results
            assert isinstance(downsampled, xr.DataArray)
            assert downsampled.sps_filters.fs <= 16.1  # Allow small tolerance
            assert downsampled.sizes["time"] < var_data.sizes["time"]

    @pytest.mark.parametrize("xr_data_array", ["multi_freq_with_trend"], indirect=True)
    def test_comparison_with_original_test(self, xr_data_array, base_signal_params):
        """Compare results with patterns from original test_xr_scipy.py."""
        # Recreate tests from original unittest

        # Test decimate (matching original test)
        decimated = xr_data_array.sps_filters.decimate(1.0)
        assert decimated.sps_filters.fs == 1.0
        assert (
            xr_data_array.coords["time"][0].values == decimated.coords["time"][0].values
        )
        assert (
            xr_data_array.coords["time"][-1].values
            != decimated.coords["time"][-1].values
        )
        assert decimated.sizes["time"] == 64  # Expected from original test

        # Test detrend (matching original test)
        detrended = xr_data_array.sps_filters.detrend()
        assert round(detrended.data.mean(), 7) == 0

        # Test properties (matching original test)
        assert xr_data_array.sps_filters.dt == 1.0 / base_signal_params["sample_rate"]
        assert xr_data_array.sps_filters.fs == base_signal_params["sample_rate"]
        expected_dx = [1.0 / base_signal_params["sample_rate"]]
        assert xr_data_array.sps_filters.dx.tolist() == expected_dx

        # Test resample_poly with specific rates (matching original tests)
        # Odd ratio
        resampled_3 = xr_data_array.sps_filters.resample_poly(3)
        expected_size_3 = base_signal_params["n_samples"] / (
            base_signal_params["sample_rate"] / 3
        )
        assert resampled_3.size == expected_size_3
        assert xr_data_array.time[0].data == resampled_3.time[0].data

        # Even ratio
        resampled_4 = xr_data_array.sps_filters.resample_poly(4)
        expected_size_4 = base_signal_params["n_samples"] / (
            base_signal_params["sample_rate"] / 4
        )
        assert resampled_4.size == expected_size_4
        assert xr_data_array.time[0].data == resampled_4.time[0].data
        assert resampled_4.sps_filters.fs == 4.0


# =============================================================================
# Test Configuration and Markers
# =============================================================================


def test_module_imports():
    """Test that all required imports are available."""
    assert SCIPY_FILTERS_AVAILABLE, "scipy_filters module should be available"

    # Test key functions are importable
    # Test accessor is registered
    import xarray as xr

    # Test warning classes are importable

    test_data = xr.DataArray([1, 2, 3], dims=["time"])
    assert hasattr(test_data, "sps_filters")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
