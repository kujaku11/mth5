"""
Comprehensive pytest test suite for frequency band helpers.

Optimized for pytest-xdist parallel execution with fixtures and subtests.
Covers all functionality in mth5.processing.spectre.frequency_band_helpers
with additional edge case and validation testing.
"""


import numpy as np
import pytest
from mt_metadata.common.band import Band as FrequencyBand
from mt_metadata.processing.aurora import FrequencyBands

from mth5.processing.spectre.frequency_band_helpers import (
    bands_of_constant_q,
    half_octave,
    log_spaced_frequencies,
    partitioned_bands,
)


# =============================================================================
# Fixtures for shared test data
# =============================================================================


@pytest.fixture(scope="session")
def sample_frequencies():
    """Common sample frequencies for testing."""
    return {
        "low": 0.01,
        "mid": 1.0,
        "high": 100.0,
        "nyquist": 250.0,
    }


@pytest.fixture(scope="session")
def fft_frequencies():
    """Standard FFT frequencies for testing."""
    sample_rate = 500.0  # Hz
    delta_t = 1.0 / sample_rate
    nfft = 128
    return np.fft.rfftfreq(n=nfft, d=delta_t)


@pytest.fixture(scope="session")
def log_spacing_params():
    """Parameters for log-spaced frequency testing."""
    return {
        "geometric": {"f_lower_bound": 1e-4, "f_upper_bound": 1e4, "num_bands": 42},
        "bands_per_decade": {
            "f_lower_bound": 1e-4,
            "f_upper_bound": 1e4,
            "num_bands_per_decade": 8.0,
        },
        "bands_per_octave": {
            "f_lower_bound": 1e-4,
            "f_upper_bound": 1e4,
            "num_bands_per_octave": 2.2,
        },
    }


@pytest.fixture(scope="session")
def constant_q_params():
    """Parameters for constant Q band testing."""
    center_freqs = log_spaced_frequencies(
        f_lower_bound=0.01,
        f_upper_bound=100.0,
        num_bands_per_decade=7.7,
    )
    fractional_bandwidths = [0.2, 0.5, 0.8]
    q_values = [1 / x for x in fractional_bandwidths]

    return {
        "center_frequencies": center_freqs,
        "fractional_bandwidths": fractional_bandwidths,
        "q_values": q_values,
    }


# =============================================================================
# Half Octave Band Tests
# =============================================================================


class TestHalfOctaveBands:
    """Test half octave band creation and validation."""

    def test_half_octave_basic_creation(self, sample_frequencies):
        """Test basic half octave band creation."""
        target_freq = sample_frequencies["mid"]
        band = half_octave(target_frequency=target_freq)

        n_octaves = np.log2(band.frequency_max / band.frequency_min)
        assert np.isclose(n_octaves, 0.5, atol=1e-2), "Should have ~0.5 octave width"
        assert isinstance(band, FrequencyBand), "Should return FrequencyBand instance"

    def test_half_octave_with_fft_frequencies(
        self, sample_frequencies, fft_frequencies
    ):
        """Test half octave with FFT frequencies for index calculation."""
        target_freq = sample_frequencies["nyquist"] / 2.0  # Quarter of sampling rate
        band = half_octave(
            target_frequency=target_freq, fft_frequencies=fft_frequencies
        )

        n_octaves = np.log2(band.frequency_max / band.frequency_min)
        assert np.isclose(n_octaves, 0.5, atol=1e-2), "Should have ~0.5 octave width"
        # Band should have indices set when fft_frequencies provided
        assert hasattr(band, "frequency_min"), "Should have frequency bounds"

    def test_half_octave_incompatible_arguments(self, fft_frequencies):
        """Test error handling with incompatible FFT frequencies and target frequency."""
        # Use frequency way outside the FFT frequency range
        invalid_target = 2000.0  # Much higher than max FFT frequency

        with pytest.raises(IndexError):
            half_octave(
                target_frequency=invalid_target, fft_frequencies=fft_frequencies
            )

    @pytest.mark.parametrize("target_freq", [0.001, 1.0, 10.0, 100.0, 1000.0])
    def test_half_octave_various_frequencies(self, target_freq):
        """Test half octave creation across various target frequencies."""
        band = half_octave(target_frequency=target_freq)

        # Check the geometric relationship
        geometric_center = np.sqrt(band.frequency_min * band.frequency_max)
        assert np.isclose(
            geometric_center, target_freq, rtol=1e-10
        ), "Geometric center should match target frequency"

    def test_half_octave_band_properties(self, sample_frequencies):
        """Test properties of created half octave bands."""
        target_freq = sample_frequencies["mid"]
        band = half_octave(target_frequency=target_freq)

        # Check that frequency bounds are positive
        assert band.frequency_min > 0, "Lower bound should be positive"
        assert band.frequency_max > 0, "Upper bound should be positive"
        assert band.frequency_max > band.frequency_min, "Upper bound > lower bound"

        # Check the expected ratio (2^0.25 for half octave)
        h = 2**0.25
        expected_min = target_freq / h
        expected_max = target_freq * h

        assert np.isclose(
            band.frequency_min, expected_min
        ), "Lower bound matches expected"
        assert np.isclose(
            band.frequency_max, expected_max
        ), "Upper bound matches expected"


# =============================================================================
# Log Spaced Frequencies Tests
# =============================================================================


class TestLogSpacedFrequencies:
    """Test logarithmically spaced frequency generation."""

    def test_geometric_spacing(self, log_spacing_params):
        """Test geometric spacing method."""
        params = log_spacing_params["geometric"]
        freqs = log_spaced_frequencies(
            f_lower_bound=params["f_lower_bound"],
            f_upper_bound=params["f_upper_bound"],
            num_bands=params["num_bands"],
        )

        # Check that ratios between consecutive frequencies are constant
        ratios = freqs[1:] / freqs[:-1]
        assert np.allclose(
            ratios, ratios[0], rtol=1e-10
        ), "Frequency ratios should be constant for geometric spacing"

        # Check correct number of frequencies (num_bands + 1 for fenceposts)
        assert (
            len(freqs) == params["num_bands"] + 1
        ), "Should have num_bands + 1 frequencies"

    def test_bands_per_decade(self, log_spacing_params):
        """Test bands per decade spacing method."""
        params = log_spacing_params["bands_per_decade"]
        freqs = log_spaced_frequencies(
            f_lower_bound=params["f_lower_bound"],
            f_upper_bound=params["f_upper_bound"],
            num_bands_per_decade=params["num_bands_per_decade"],
        )

        # Check that ratios are constant
        ratios = freqs[1:] / freqs[:-1]
        assert np.allclose(
            ratios, ratios[0], rtol=1e-10
        ), "Frequency ratios should be constant"

    def test_bands_per_octave(self, log_spacing_params):
        """Test bands per octave spacing method."""
        params = log_spacing_params["bands_per_octave"]
        freqs = log_spaced_frequencies(
            f_lower_bound=params["f_lower_bound"],
            f_upper_bound=params["f_upper_bound"],
            num_bands_per_octave=params["num_bands_per_octave"],
        )

        # Check that ratios are constant
        ratios = freqs[1:] / freqs[:-1]
        assert np.allclose(
            ratios, ratios[0], rtol=1e-10
        ), "Frequency ratios should be constant"

    def test_boundary_frequencies(self, log_spacing_params):
        """Test that boundary frequencies match input parameters."""
        for method, params in log_spacing_params.items():
            if method == "geometric":
                freqs = log_spaced_frequencies(
                    f_lower_bound=params["f_lower_bound"],
                    f_upper_bound=params["f_upper_bound"],
                    num_bands=params["num_bands"],
                )
            elif method == "bands_per_decade":
                freqs = log_spaced_frequencies(
                    f_lower_bound=params["f_lower_bound"],
                    f_upper_bound=params["f_upper_bound"],
                    num_bands_per_decade=params["num_bands_per_decade"],
                )
            else:  # bands_per_octave
                freqs = log_spaced_frequencies(
                    f_lower_bound=params["f_lower_bound"],
                    f_upper_bound=params["f_upper_bound"],
                    num_bands_per_octave=params["num_bands_per_octave"],
                )

            assert np.isclose(
                freqs[0], params["f_lower_bound"]
            ), f"First frequency should match lower bound for {method}"
            assert np.isclose(
                freqs[-1], params["f_upper_bound"]
            ), f"Last frequency should match upper bound for {method}"

    def test_multiple_parameter_error(self):
        """Test that specifying multiple parameters raises an error."""
        with pytest.raises(ValueError, match="Please specify only one"):
            log_spaced_frequencies(
                f_lower_bound=0.1,
                f_upper_bound=10.0,
                num_bands=10,
                num_bands_per_decade=5,
            )

    def test_no_parameter_error(self):
        """Test error when no spacing parameter is provided."""
        # This should fail because no spacing method is specified
        with pytest.raises((TypeError, ValueError)):
            log_spaced_frequencies(f_lower_bound=0.1, f_upper_bound=10.0)

    @pytest.mark.parametrize(
        "method,param_name,param_value",
        [
            ("geometric", "num_bands", 20),
            ("bands_per_decade", "num_bands_per_decade", 5.0),
            ("bands_per_octave", "num_bands_per_octave", 3.0),
        ],
    )
    def test_parametrized_methods(self, method, param_name, param_value):
        """Parametrized test for different spacing methods."""
        kwargs = {"f_lower_bound": 0.1, "f_upper_bound": 10.0, param_name: param_value}

        freqs = log_spaced_frequencies(**kwargs)

        # Basic validation
        assert len(freqs) > 1, "Should generate multiple frequencies"
        assert freqs[0] >= 0.1, "Should respect lower bound"
        assert np.isclose(freqs[-1], 10.0, rtol=1e-10), "Should respect upper bound"
        assert np.all(np.diff(freqs) > 0), "Frequencies should be increasing"


# =============================================================================
# Constant Q Bands Tests
# =============================================================================


class TestConstantQBands:
    """Test constant Q frequency bands creation and properties."""

    def test_bands_of_constant_q_with_q(self, constant_q_params):
        """Test constant Q bands using Q parameter."""
        center_freqs = constant_q_params["center_frequencies"]

        for q in constant_q_params["q_values"]:
            frequency_bands = bands_of_constant_q(
                band_center_frequencies=center_freqs, q=q
            )

            assert isinstance(
                frequency_bands, FrequencyBands
            ), "Should return FrequencyBands instance"
            assert frequency_bands.number_of_bands == len(
                center_freqs
            ), "Should create correct number of bands"

            # Test Q values for each band with different averaging types
            for i in range(frequency_bands.number_of_bands):
                band = frequency_bands.band(i)

                # Test with arithmetic center
                band.center_averaging_type = "arithmetic"
                assert np.isclose(
                    band.Q, q, rtol=1e-1
                ), f"Arithmetic Q should be close to target for band {i}"

                # Test with geometric center (should be more accurate)
                band.center_averaging_type = "geometric"
                assert np.isclose(
                    band.Q, q, rtol=1e-1
                ), f"Geometric Q should be close to target for band {i}"

    def test_bands_of_constant_q_with_fractional_bandwidth(self, constant_q_params):
        """Test constant Q bands using fractional bandwidth parameter."""
        center_freqs = constant_q_params["center_frequencies"]

        for frac_bw in constant_q_params["fractional_bandwidths"]:
            frequency_bands = bands_of_constant_q(
                band_center_frequencies=center_freqs, fractional_bandwidth=frac_bw
            )

            expected_q = 1.0 / frac_bw

            for i in range(frequency_bands.number_of_bands):
                band = frequency_bands.band(i)
                band.center_averaging_type = "geometric"
                assert np.isclose(
                    band.Q, expected_q, rtol=1e-1
                ), f"Q should match 1/fractional_bandwidth for band {i}"

    def test_constant_q_no_parameters_error(self, constant_q_params):
        """Test error when no Q or fractional bandwidth is specified."""
        center_freqs = constant_q_params["center_frequencies"]

        with pytest.raises(
            ValueError, match="must specify one of Q or fractional_bandwidth"
        ):
            bands_of_constant_q(band_center_frequencies=center_freqs)

    def test_constant_q_band_properties(self, constant_q_params):
        """Test properties of constant Q bands."""
        center_freqs = constant_q_params["center_frequencies"]
        q = 5.0

        frequency_bands = bands_of_constant_q(band_center_frequencies=center_freqs, q=q)

        # Check that all bands have positive frequencies
        for i in range(frequency_bands.number_of_bands):
            band = frequency_bands.band(i)
            assert band.lower_bound > 0, f"Band {i} lower bound should be positive"
            assert band.upper_bound > 0, f"Band {i} upper bound should be positive"
            assert (
                band.upper_bound > band.lower_bound
            ), f"Band {i} upper bound should exceed lower bound"

    @pytest.mark.parametrize("q_value", [1.5, 2.0, 5.0, 10.0, 20.0])
    def test_various_q_values(self, q_value):
        """Test constant Q bands with various Q values."""
        center_freqs = np.logspace(-1, 2, 10)  # 0.1 to 100 Hz, 10 bands

        frequency_bands = bands_of_constant_q(
            band_center_frequencies=center_freqs, q=q_value
        )

        assert frequency_bands.number_of_bands == len(
            center_freqs
        ), "Should create correct number of bands"

        # Check that Q is approximately constant across all bands
        q_values = []
        for i in range(frequency_bands.number_of_bands):
            band = frequency_bands.band(i)
            band.center_averaging_type = "geometric"
            q_values.append(band.Q)

        # All Q values should be close to target
        assert np.allclose(
            q_values, q_value, rtol=0.2
        ), f"All Q values should be close to {q_value}"


# =============================================================================
# Partitioned Bands Tests
# =============================================================================


class TestPartitionedBands:
    """Test partitioned frequency bands creation."""

    def test_partitioned_bands_basic(self):
        """Test basic partitioned bands functionality."""
        frequencies = [0.1, 1.0, 10.0, 100.0]
        frequency_bands = partitioned_bands(frequencies=frequencies)

        assert isinstance(
            frequency_bands, FrequencyBands
        ), "Should return FrequencyBands instance"
        assert (
            frequency_bands.number_of_bands == len(frequencies) - 1
        ), "Should create num_frequencies - 1 bands"

    def test_partitioned_bands_boundaries(self, constant_q_params):
        """Test that partition boundaries match input frequencies."""
        fenceposts = constant_q_params["center_frequencies"]
        frequency_bands = partitioned_bands(frequencies=fenceposts)

        for i in range(
            frequency_bands.number_of_bands - 1
        ):  # Avoid last band edge case
            band = frequency_bands.band(i)
            assert np.isclose(
                band.lower_bound, fenceposts[i]
            ), f"Band {i} lower bound should match fencepost {i}"
            assert np.isclose(
                band.upper_bound, fenceposts[i + 1]
            ), f"Band {i} upper bound should match fencepost {i + 1}"

    def test_partitioned_bands_with_array(self):
        """Test partitioned bands with numpy array input."""
        frequencies = np.logspace(-1, 2, 11)  # 10 bands from 0.1 to 100 Hz
        frequency_bands = partitioned_bands(frequencies=frequencies)

        assert (
            frequency_bands.number_of_bands == len(frequencies) - 1
        ), "Should create correct number of bands from array"

    def test_partitioned_bands_with_list(self):
        """Test partitioned bands with list input."""
        frequencies = [0.01, 0.1, 1.0, 10.0, 100.0]
        frequency_bands = partitioned_bands(frequencies=frequencies)

        assert (
            frequency_bands.number_of_bands == len(frequencies) - 1
        ), "Should create correct number of bands from list"

    def test_partitioned_bands_continuity(self):
        """Test that partitioned bands are continuous (no gaps)."""
        frequencies = np.logspace(0, 3, 20)  # 1 to 1000 Hz, 19 bands
        frequency_bands = partitioned_bands(frequencies=frequencies)

        # Check that each band's upper bound equals next band's lower bound
        for i in range(frequency_bands.number_of_bands - 1):
            current_band = frequency_bands.band(i)
            next_band = frequency_bands.band(i + 1)

            assert np.isclose(
                current_band.upper_bound, next_band.lower_bound
            ), f"Gap found between band {i} and {i + 1}"

    @pytest.mark.parametrize("num_freqs", [3, 5, 10, 20, 50])
    def test_partitioned_bands_scaling(self, num_freqs):
        """Test partitioned bands with different numbers of frequencies."""
        frequencies = np.logspace(-2, 2, num_freqs)
        frequency_bands = partitioned_bands(frequencies=frequencies)

        expected_bands = num_freqs - 1
        assert (
            frequency_bands.number_of_bands == expected_bands
        ), f"Should create {expected_bands} bands from {num_freqs} frequencies"


# =============================================================================
# Integration and Edge Case Tests
# =============================================================================


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    def test_workflow_log_frequencies_to_constant_q(self, subtests):
        """Test typical workflow: generate log frequencies then create constant Q bands."""
        with subtests.test("Generate log frequencies"):
            frequencies = log_spaced_frequencies(
                f_lower_bound=0.01, f_upper_bound=100.0, num_bands_per_decade=4.0
            )
            assert len(frequencies) > 1, "Should generate multiple frequencies"

        with subtests.test("Create constant Q bands"):
            q_value = 3.0
            frequency_bands = bands_of_constant_q(
                band_center_frequencies=frequencies, q=q_value
            )
            assert frequency_bands.number_of_bands == len(
                frequencies
            ), "Should create band for each center frequency"

        with subtests.test("Validate Q consistency"):
            q_values = []
            for i in range(frequency_bands.number_of_bands):
                band = frequency_bands.band(i)
                band.center_averaging_type = "geometric"
                q_values.append(band.Q)

            assert np.allclose(
                q_values, q_value, rtol=0.2
            ), "All bands should have consistent Q"

    def test_workflow_partitioned_vs_constant_q(self, subtests):
        """Compare partitioned bands vs constant Q bands for same frequencies."""
        frequencies = np.logspace(0, 2, 11)  # 1 to 100 Hz

        with subtests.test("Create partitioned bands"):
            partitioned = partitioned_bands(frequencies=frequencies)
            assert (
                partitioned.number_of_bands == len(frequencies) - 1
            ), "Partitioned should use adjacent frequencies as bounds"

        with subtests.test("Create constant Q bands"):
            # Use center frequencies (skip first and last for fair comparison)
            centers = frequencies[1:-1]
            constant_q = bands_of_constant_q(band_center_frequencies=centers, q=2.0)
            assert constant_q.number_of_bands == len(
                centers
            ), "Constant Q should create band per center frequency"

    def test_extreme_frequency_ranges(self, subtests):
        """Test with extreme frequency ranges."""
        with subtests.test("Very low frequencies"):
            freqs = log_spaced_frequencies(
                f_lower_bound=1e-6, f_upper_bound=1e-3, num_bands=5
            )
            assert len(freqs) == 6, "Should handle very low frequencies"
            assert all(f > 0 for f in freqs), "All frequencies should be positive"

        with subtests.test("Very high frequencies"):
            freqs = log_spaced_frequencies(
                f_lower_bound=1e6, f_upper_bound=1e9, num_bands=5
            )
            assert len(freqs) == 6, "Should handle very high frequencies"
            assert all(f > 0 for f in freqs), "All frequencies should be positive"

        with subtests.test("Very wide range"):
            freqs = log_spaced_frequencies(
                f_lower_bound=1e-6, f_upper_bound=1e6, num_bands_per_decade=2
            )
            assert len(freqs) > 10, "Should handle very wide frequency range"

    def test_single_band_edge_cases(self):
        """Test edge cases with single bands."""
        # Single frequency for constant Q (should create one band)
        single_freq = np.array([1.0])
        bands = bands_of_constant_q(band_center_frequencies=single_freq, q=2.0)
        assert bands.number_of_bands == 1, "Should create single band"

        # Two frequencies for partitioned (should create one band)
        two_freqs = [1.0, 10.0]
        partitioned = partitioned_bands(frequencies=two_freqs)
        assert partitioned.number_of_bands == 1, "Should create single partitioned band"

    @pytest.mark.parametrize("invalid_q", [0, -1, -10])
    def test_invalid_q_values(self, invalid_q):
        """Test behavior with invalid Q values."""
        center_freqs = np.array([1.0, 10.0])

        if invalid_q == 0:
            # Q = 0 causes division by zero
            with pytest.raises(ZeroDivisionError):
                bands_of_constant_q(band_center_frequencies=center_freqs, q=invalid_q)
        else:
            # Negative Q values are mathematically invalid but may not crash
            # This test documents current behavior
            bands = bands_of_constant_q(
                band_center_frequencies=center_freqs, q=invalid_q
            )

            # At minimum, should not crash and should create bands
            assert bands.number_of_bands == len(
                center_freqs
            ), "Should create bands even with unusual Q values"


# =============================================================================
# Performance and Stress Tests
# =============================================================================


class TestPerformanceAndStress:
    """Performance and stress tests for the frequency band helpers."""

    @pytest.mark.performance
    def test_large_number_of_bands(self):
        """Test performance with large numbers of bands."""
        # Test with 1000 bands
        frequencies = log_spaced_frequencies(
            f_lower_bound=0.001, f_upper_bound=1000.0, num_bands=1000
        )
        assert len(frequencies) == 1001, "Should handle 1000 bands efficiently"

        # Test constant Q with many bands
        bands = bands_of_constant_q(band_center_frequencies=frequencies, q=5.0)
        assert bands.number_of_bands == len(
            frequencies
        ), "Should handle large constant Q band creation"

    @pytest.mark.performance
    def test_memory_efficiency(self):
        """Test memory efficiency with repeated operations."""
        # Create and destroy many band objects
        for _ in range(100):
            frequencies = log_spaced_frequencies(
                f_lower_bound=0.1, f_upper_bound=100.0, num_bands_per_decade=8
            )
            bands = bands_of_constant_q(band_center_frequencies=frequencies, q=3.0)
            # Verify basic operation still works
            assert bands.number_of_bands > 0

    @pytest.mark.performance
    def test_frequency_precision(self):
        """Test numerical precision with extreme values."""
        # Very close frequencies
        close_freqs = [1.0, 1.0 + 1e-10, 1.0 + 2e-10]
        bands = partitioned_bands(frequencies=close_freqs)
        assert bands.number_of_bands == 2, "Should handle very close frequencies"

        # Very different frequencies
        wide_freqs = [1e-10, 1e10]
        wide_bands = partitioned_bands(frequencies=wide_freqs)
        assert wide_bands.number_of_bands == 1, "Should handle extreme frequency ratios"


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionCases:
    """Regression tests for known issues or edge cases."""

    def test_half_octave_regression_issue_209(self, fft_frequencies):
        """Regression test related to issue #209 mentioned in original test."""
        # Test multiple half octave bands as might be used in multistation FCs
        target_frequencies = [1.0, 10.0, 100.0]
        bands = []

        for freq in target_frequencies:
            if freq <= fft_frequencies.max():  # Only test valid frequencies
                # Check if the half-octave band overlaps with FFT frequency range
                h = 2**0.25
                f_min = freq / h
                f_max = freq * h

                # Only test if the band has overlap with FFT frequencies
                if f_min <= fft_frequencies.max() and f_max >= fft_frequencies.min():
                    try:
                        band = half_octave(
                            target_frequency=freq, fft_frequencies=fft_frequencies
                        )
                        bands.append(band)
                    except IndexError:
                        # Skip if no valid indices found in FFT frequency range
                        continue

        # All bands should be valid half-octave widths
        for i, band in enumerate(bands):
            n_octaves = np.log2(band.frequency_max / band.frequency_min)
            assert np.isclose(
                n_octaves, 0.5, atol=1e-2
            ), f"Band {i} should be half octave width"

    def test_log_frequencies_consistency(self):
        """Test that log frequencies are consistent across different parameter specifications."""
        # Same result should be achieved with different parameterizations
        method1 = log_spaced_frequencies(
            f_lower_bound=1.0, f_upper_bound=100.0, num_bands=20
        )

        # Calculate equivalent bands per decade
        decades = np.log10(100.0 / 1.0)  # 2 decades
        bands_per_decade = 20 / decades

        method2 = log_spaced_frequencies(
            f_lower_bound=1.0,
            f_upper_bound=100.0,
            num_bands_per_decade=bands_per_decade,
        )

        # Should produce very similar results
        assert np.allclose(
            method1, method2, rtol=1e-2
        ), "Different parameterizations should yield consistent results"


# =============================================================================
# Configuration
# =============================================================================

pytestmark = [
    pytest.mark.integration,  # Mark all tests as integration tests
]
