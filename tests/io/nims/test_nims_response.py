# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for NIMS Response class testing.

Created on Fri Nov 22 17:30:00 2024

@author: jpeacock

Pytest translation and expansion of test_nims_response.py with additional
coverage for missing functionality.

Status: Comprehensive test coverage including:
- Response initialization and configuration
- Filter property generation (magnetic/electric)
- Hardware-specific filter selection
- Time delay filter generation
- Channel response construction
- Error handling and edge cases
- Filter parameter validation
"""

from collections import OrderedDict

import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest
from mt_metadata.timeseries.filters import (
    ChannelResponse,
    CoefficientFilter,
    PoleZeroFilter,
    TimeDelayFilter,
)

from mth5.io.nims import Response
from mth5.io.nims.response_filters import ResponseError


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def response():
    """Default Response object for testing."""
    return Response()


@pytest.fixture
def response_with_system_id():
    """Response object with system ID set."""
    return Response(system_id="test_system_001")


@pytest.fixture
def response_custom_params():
    """Response object with custom parameters."""
    return Response(
        system_id="custom_001",
        hardware="HP200",
        instrument_type="custom",
        sample_rate=8,
        e_conversion_factor=1000000.0,
        h_conversion_factor=50,
    )


@pytest.fixture
def expected_pc_high_pass():
    """Expected PC high pass filter dictionary."""
    return OrderedDict(
        [
            ("gain", 1.0),
            ("name", "nims_1_pole_butterworth"),
            ("normalization_factor", 1.0),
            ("poles", np.array([-3.333333e-05 + 0.0j])),
            ("sequence_number", 0),
            ("type", "zpk"),
            ("units_in", "Volt"),
            ("units_out", "Volt"),
            ("zeros", np.array([0.0 + 0.0j])),
        ]
    )


@pytest.fixture
def expected_hp_high_pass():
    """Expected HP high pass filter dictionary."""
    return OrderedDict(
        [
            ("gain", 1.0),
            ("name", "nims_1_pole_butterworth"),
            ("normalization_factor", 1.0),
            ("poles", np.array([-1.66667e-04 + 0.0j])),
            ("sequence_number", 0),
            ("type", "zpk"),
            ("units_in", "Volt"),
            ("units_out", "Volt"),
            ("zeros", np.array([0.0 + 0.0j])),
        ]
    )


@pytest.fixture
def expected_dipole_filter():
    """Expected dipole filter dictionary."""
    return OrderedDict(
        [
            ("gain", 100.0),
            ("name", "dipole_100.00"),
            ("sequence_number", 0),
            ("type", "coefficient"),
            ("units_in", "Volt per meter"),
            ("units_out", "Volt"),
        ]
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestResponseInitialization:
    """Test Response object initialization and configuration."""

    def test_default_initialization(self, response):
        """Test default Response initialization."""
        assert response.system_id is None
        assert response.hardware == "PC"
        assert response.instrument_type == "backbone"
        assert response.sample_rate == 1
        assert response.e_conversion_factor == 409600000.0
        assert response.h_conversion_factor == 100

    def test_initialization_with_system_id(self, response_with_system_id):
        """Test Response initialization with system ID."""
        assert response_with_system_id.system_id == "test_system_001"

    def test_initialization_with_kwargs(self, response_custom_params):
        """Test Response initialization with custom parameters."""
        assert response_custom_params.system_id == "custom_001"
        assert response_custom_params.hardware == "HP200"
        assert response_custom_params.instrument_type == "custom"
        assert response_custom_params.sample_rate == 8
        assert response_custom_params.e_conversion_factor == 1000000.0
        assert response_custom_params.h_conversion_factor == 50

    def test_time_delays_dict_structure(self, response):
        """Test time delays dictionary structure."""
        expected_keys = ["hp200", 1, 8]
        assert list(response.time_delays_dict.keys()) == expected_keys

        # Check each sample rate has required channels
        for sample_rate in expected_keys:
            channels = response.time_delays_dict[sample_rate]
            assert "hx" in channels
            assert "hy" in channels
            assert "hz" in channels
            assert "ex" in channels
            assert "ey" in channels

    def test_dynamic_attribute_setting(self):
        """Test dynamic attribute setting via kwargs."""
        response = Response(
            custom_attr="test_value", numeric_attr=42, list_attr=[1, 2, 3]
        )
        assert response.custom_attr == "test_value"
        assert response.numeric_attr == 42
        assert response.list_attr == [1, 2, 3]


class TestMagneticFilters:
    """Test magnetic filter properties and methods."""

    def test_magnetic_low_pass_filter(self, response):
        """Test magnetic low pass filter generation."""
        filter_obj = response.magnetic_low_pass

        assert isinstance(filter_obj, PoleZeroFilter)
        assert filter_obj.name == "nims_3_pole_butterworth"
        assert filter_obj.units_in == "nanoTesla"
        assert filter_obj.units_out == "Volt"
        assert len(filter_obj.zeros) == 0
        assert len(filter_obj.poles) == 3
        assert filter_obj.normalization_factor == 2002.26936395594

        # Check specific pole values
        expected_poles = [
            complex(-6.283185, 10.882477),
            complex(-6.283185, -10.882477),
            complex(-12.566371, 0),
        ]
        np.testing.assert_array_almost_equal(filter_obj.poles, expected_poles)

    def test_magnetic_conversion_filter(self, response):
        """Test magnetic conversion filter generation."""
        filter_obj = response.magnetic_conversion

        assert isinstance(filter_obj, CoefficientFilter)
        assert filter_obj.name == "h_analog_to_digital"
        assert filter_obj.gain == response.h_conversion_factor
        assert filter_obj.units_in == "Volt"
        assert filter_obj.units_out == "digital counts"

    def test_magnetic_conversion_custom_factor(self, response_custom_params):
        """Test magnetic conversion with custom conversion factor."""
        filter_obj = response_custom_params.magnetic_conversion
        assert filter_obj.gain == 50  # Custom h_conversion_factor

    def test_get_magnetic_filter_list(self, response):
        """Test complete magnetic filter chain."""
        filters = response._get_magnetic_filter("hx")

        assert len(filters) == 3
        assert isinstance(filters[0], PoleZeroFilter)  # Low pass
        assert isinstance(filters[1], CoefficientFilter)  # Conversion
        assert isinstance(filters[2], TimeDelayFilter)  # Time delay

        # Check filter sequence
        assert filters[0].name == "nims_3_pole_butterworth"
        assert filters[1].name == "h_analog_to_digital"
        assert filters[2].name == "hx_time_offset"

    @pytest.mark.parametrize("channel", ["hx", "hy", "hz"])
    def test_magnetic_filter_different_channels(self, response, channel):
        """Test magnetic filter generation for different channels."""
        filters = response._get_magnetic_filter(channel)

        assert len(filters) == 3
        # Time delay filter should have channel-specific name
        assert filters[2].name == f"{channel}_time_offset"


class TestElectricFilters:
    """Test electric filter properties and methods."""

    def test_electric_low_pass_filter(self, response):
        """Test electric low pass filter generation."""
        filter_obj = response.electric_low_pass

        assert isinstance(filter_obj, PoleZeroFilter)
        assert filter_obj.name == "nims_5_pole_butterworth"
        assert filter_obj.units_in == "Volt"
        assert filter_obj.units_out == "Volt"
        assert len(filter_obj.zeros) == 0
        assert len(filter_obj.poles) == 5
        assert filter_obj.normalization_factor == 313383.493219835

        # Check specific pole values
        expected_poles = [
            complex(-3.88301, 11.9519),
            complex(-3.88301, -11.9519),
            complex(-10.1662, 7.38651),
            complex(-10.1662, -7.38651),
            complex(-12.5664, 0.0),
        ]
        np.testing.assert_array_almost_equal(filter_obj.poles, expected_poles)

    def test_electric_high_pass_pc(self, response):
        """Test PC electric high pass filter."""
        filter_obj = response.electric_high_pass_pc

        assert isinstance(filter_obj, PoleZeroFilter)
        assert filter_obj.name == "nims_1_pole_butterworth"
        assert filter_obj.normalization_factor == 1
        assert len(filter_obj.zeros) == 1
        assert len(filter_obj.poles) == 1

        np.testing.assert_array_almost_equal(filter_obj.zeros, [complex(0.0, 0.0)])
        np.testing.assert_array_almost_equal(
            filter_obj.poles, [complex(-3.333333e-05, 0.0)]
        )

    def test_electric_high_pass_hp(self, response):
        """Test HP electric high pass filter."""
        filter_obj = response.electric_high_pass_hp

        assert isinstance(filter_obj, PoleZeroFilter)
        assert filter_obj.name == "nims_1_pole_butterworth"
        assert filter_obj.normalization_factor == 1
        assert len(filter_obj.zeros) == 1
        assert len(filter_obj.poles) == 1

        np.testing.assert_array_almost_equal(filter_obj.zeros, [complex(0.0, 0.0)])
        np.testing.assert_array_almost_equal(
            filter_obj.poles, [complex(-1.66667e-04, 0.0)]
        )

    def test_electric_conversion_filter(self, response):
        """Test electric conversion filter generation."""
        filter_obj = response.electric_conversion

        assert isinstance(filter_obj, CoefficientFilter)
        assert filter_obj.name == "e_analog_to_digital"
        assert filter_obj.gain == response.e_conversion_factor
        assert filter_obj.units_in == "Volt"
        assert filter_obj.units_out == "digital counts"

    def test_electric_physical_units_filter(self, response):
        """Test electric physical units conversion filter."""
        filter_obj = response.electric_physical_units

        assert isinstance(filter_obj, CoefficientFilter)
        assert filter_obj.name == "to_mt_units"
        assert filter_obj.gain == 1e-6
        assert filter_obj.units_in == "milliVolt per kilometer"
        assert filter_obj.units_out == "Volt per meter"

    def test_get_electric_filter_backbone(self, response):
        """Test electric filter chain for backbone instruments."""
        filters = response._get_electric_filter("ex", 100.0)

        assert len(filters) == 6
        assert isinstance(filters[0], CoefficientFilter)  # Physical units
        assert isinstance(filters[1], CoefficientFilter)  # Dipole
        assert isinstance(filters[2], PoleZeroFilter)  # Low pass
        assert isinstance(filters[3], PoleZeroFilter)  # High pass
        assert isinstance(filters[4], CoefficientFilter)  # Conversion
        assert isinstance(filters[5], TimeDelayFilter)  # Time delay

        # Check filter names
        assert filters[0].name == "to_mt_units"
        assert filters[1].name == "dipole_100.00"
        assert filters[2].name == "nims_5_pole_butterworth"
        assert filters[3].name == "nims_1_pole_butterworth"
        assert filters[4].name == "e_analog_to_digital"
        assert filters[5].name == "ex_time_offset"

    def test_get_electric_filter_non_backbone(self):
        """Test electric filter chain for non-backbone instruments."""
        response = Response(instrument_type="custom")
        filters = response._get_electric_filter("ex", 100.0)

        # Should have one less filter (no high pass)
        assert len(filters) == 5
        filter_names = [f.name for f in filters]
        assert "nims_1_pole_butterworth" not in filter_names


class TestHighPassFilters:
    """Test hardware-specific high pass filter selection."""

    def test_get_electric_high_pass_pc(self, response, expected_pc_high_pass):
        """Test PC high pass filter selection."""
        filter_obj = response.get_electric_high_pass(hardware="pc")
        assert filter_obj.to_dict(single=True) == expected_pc_high_pass

    def test_get_electric_high_pass_pc_case_insensitive(
        self, response, expected_pc_high_pass
    ):
        """Test PC high pass filter selection with case variations."""
        test_cases = ["PC", "pc", "Pc", "pC"]
        for hardware in test_cases:
            filter_obj = response.get_electric_high_pass(hardware=hardware)
            assert filter_obj.to_dict(single=True) == expected_pc_high_pass

    def test_get_electric_high_pass_hp(self, response, expected_hp_high_pass):
        """Test HP high pass filter selection."""
        filter_obj = response.get_electric_high_pass(hardware="HP")
        assert filter_obj.to_dict(single=True) == expected_hp_high_pass

    def test_get_electric_high_pass_hp_case_insensitive(
        self, response, expected_hp_high_pass
    ):
        """Test HP high pass filter selection with case variations."""
        test_cases = ["HP", "hp", "Hp", "hP", "HP200", "hp200"]
        for hardware in test_cases:
            filter_obj = response.get_electric_high_pass(hardware=hardware)
            assert filter_obj.to_dict(single=True) == expected_hp_high_pass

    def test_get_electric_high_pass_invalid_hardware(self, response):
        """Test high pass filter with invalid hardware."""
        with pytest.raises(
            ResponseError, match="Hardware value invalid not understood"
        ):
            response.get_electric_high_pass(hardware="invalid")

    def test_get_electric_high_pass_sets_hardware_attribute(self, response):
        """Test that get_electric_high_pass sets hardware attribute."""
        original_hardware = response.hardware
        response.get_electric_high_pass(hardware="HP200")
        assert response.hardware == "HP200"
        assert response.hardware != original_hardware


class TestDipoleFilter:
    """Test dipole filter generation."""

    def test_dipole_filter_standard(self, response, expected_dipole_filter):
        """Test dipole filter with standard length."""
        filter_obj = response.dipole_filter(100)
        assert filter_obj.to_dict(single=True) == expected_dipole_filter

    @pytest.mark.parametrize(
        "length,expected_name",
        [
            (50, "dipole_50.00"),
            (75.5, "dipole_75.50"),
            (123.456, "dipole_123.46"),
            (1, "dipole_1.00"),
            (0.5, "dipole_0.50"),
        ],
    )
    def test_dipole_filter_various_lengths(self, response, length, expected_name):
        """Test dipole filter with various lengths."""
        filter_obj = response.dipole_filter(length)

        assert isinstance(filter_obj, CoefficientFilter)
        assert filter_obj.name == expected_name
        assert filter_obj.gain == length
        assert filter_obj.units_in == "Volt per meter"
        assert filter_obj.units_out == "Volt"

    def test_dipole_filter_zero_length(self, response):
        """Test dipole filter with zero length."""
        # Note: CoefficientFilter validation requires gain > 0, so we expect this to fail
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            response.dipole_filter(0)

    def test_dipole_filter_negative_length(self, response):
        """Test dipole filter with negative length."""
        # Note: CoefficientFilter validation requires gain > 0, so we expect this to fail
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            response.dipole_filter(-50)


class TestTimeDelayFilters:
    """Test time delay filter generation."""

    @pytest.mark.parametrize(
        "sample_rate,channel,expected_delay",
        [
            (1, "hx", -0.1920),
            (1, "hy", -0.2010),
            (1, "hz", -0.2100),
            (1, "ex", -0.2850),
            (1, "ey", -0.2850),
            (8, "hx", -0.2455),
            (8, "hy", -0.2365),
            (8, "hz", -0.2275),
            (8, "ex", -0.1525),
            (8, "ey", -0.1525),
            ("hp200", "hx", -0.0055),
            ("hp200", "hy", -0.0145),
            ("hp200", "hz", -0.0235),
            ("hp200", "ex", -0.1525),
            ("hp200", "ey", -0.0275),
        ],
    )
    def test_get_dt_filter_specific_values(
        self, response, sample_rate, channel, expected_delay
    ):
        """Test time delay filter with specific sample rates and channels."""
        filter_obj = response._get_dt_filter(channel, sample_rate)

        assert isinstance(filter_obj, TimeDelayFilter)
        assert filter_obj.name == f"{channel}_time_offset"
        assert filter_obj.delay == expected_delay
        assert filter_obj.units_in == "digital counts"
        assert filter_obj.units_out == "digital counts"

    def test_get_dt_filter_invalid_sample_rate(self, response):
        """Test time delay filter with invalid sample rate."""
        with pytest.raises(KeyError):
            response._get_dt_filter("hx", 999)

    def test_get_dt_filter_invalid_channel(self, response):
        """Test time delay filter with invalid channel."""
        with pytest.raises(KeyError):
            response._get_dt_filter("invalid", 1)


class TestChannelResponse:
    """Test complete channel response generation."""

    def test_get_electric_channel_response(self, response):
        """Test electric channel response generation."""
        channel_response = response.get_channel_response("ex", 100)

        assert isinstance(channel_response, ChannelResponse)
        assert len(channel_response.filters_list) == 6
        assert channel_response.units_in == "milliVolt per kilometer"
        assert channel_response.units_out == "digital counts"

    def test_get_magnetic_channel_response(self, response):
        """Test magnetic channel response generation."""
        channel_response = response.get_channel_response("hx")

        assert isinstance(channel_response, ChannelResponse)
        assert len(channel_response.filters_list) == 3
        assert channel_response.units_in == "nanoTesla"
        assert channel_response.units_out == "digital counts"

    @pytest.mark.parametrize("channel", ["ex", "ey"])
    def test_get_electric_channel_response_various_channels(self, response, channel):
        """Test electric channel response for different electric channels."""
        channel_response = response.get_channel_response(channel, 75.5)

        assert isinstance(channel_response, ChannelResponse)
        assert len(channel_response.filters_list) == 6
        # Check that dipole filter has correct length
        dipole_filter = channel_response.filters_list[1]
        assert dipole_filter.gain == 75.5

    @pytest.mark.parametrize("channel", ["hx", "hy", "hz"])
    def test_get_magnetic_channel_response_various_channels(self, response, channel):
        """Test magnetic channel response for different magnetic channels."""
        channel_response = response.get_channel_response(channel)

        assert isinstance(channel_response, ChannelResponse)
        assert len(channel_response.filters_list) == 3

    def test_get_channel_response_unsupported_channel(self, response):
        """Test channel response with unsupported channel type."""
        with pytest.raises(ValueError, match="Channel invalid not supported"):
            response.get_channel_response("invalid")

    def test_get_channel_response_unsupported_magnetic_channel(self, response):
        """Test magnetic channel response with unsupported channels (no time delay data)."""
        # Channels like bx, by, bz don't have time delay data in the time_delays_dict
        with pytest.raises(KeyError):
            response.get_channel_response("bx")

    def test_get_channel_response_default_dipole_length(self, response):
        """Test electric channel response with default dipole length."""
        channel_response = response.get_channel_response("ex")

        # Check default dipole length of 1
        dipole_filter = channel_response.filters_list[1]
        assert dipole_filter.gain == 1.0
        assert dipole_filter.name == "dipole_1.00"


class TestFilterIntegration:
    """Test integration between different filter components."""

    def test_electric_filter_chain_units(self, response):
        """Test that electric filter chain has consistent units."""
        filters = response._get_electric_filter("ex", 100.0)

        # Check unit progression through filter chain
        assert filters[0].units_in == "milliVolt per kilometer"
        assert filters[0].units_out == "Volt per meter"
        assert filters[1].units_in == "Volt per meter"
        assert filters[1].units_out == "Volt"
        # Subsequent filters should maintain Volt → Volt → digital counts → digital counts
        assert filters[-1].units_out == "digital counts"

    def test_magnetic_filter_chain_units(self, response):
        """Test that magnetic filter chain has consistent units."""
        filters = response._get_magnetic_filter("hx")

        # Check unit progression
        assert filters[0].units_in == "nanoTesla"
        assert filters[0].units_out == "Volt"
        assert filters[1].units_in == "Volt"
        assert filters[1].units_out == "digital counts"
        assert filters[2].units_in == "digital counts"
        assert filters[2].units_out == "digital counts"

    def test_filter_sequence_numbers(self, response):
        """Test that filters have appropriate sequence numbers."""
        channel_response = response.get_channel_response("ex", 100)

        # Check that sequence numbers are set (should be 0 for individual filters)
        for i, filter_obj in enumerate(channel_response.filters_list):
            assert hasattr(filter_obj, "sequence_number")


class TestResponseError:
    """Test ResponseError exception."""

    def test_response_error_inheritance(self):
        """Test that ResponseError inherits from Exception."""
        assert issubclass(ResponseError, Exception)

    def test_response_error_message(self):
        """Test ResponseError with custom message."""
        error_msg = "Test error message"
        error = ResponseError(error_msg)
        assert str(error) == error_msg


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_response_with_extreme_conversion_factors(self):
        """Test Response with extreme conversion factors."""
        response = Response(e_conversion_factor=1e-10, h_conversion_factor=1e10)

        e_filter = response.electric_conversion
        h_filter = response.magnetic_conversion

        assert e_filter.gain == 1e-10
        assert h_filter.gain == 1e10

    def test_response_sample_rate_effects(self):
        """Test that sample rate affects time delay filters."""
        response = Response(sample_rate=8)

        dt_filter_8hz = response._get_dt_filter("hx", 8)
        dt_filter_1hz = response._get_dt_filter("hx", 1)

        assert dt_filter_8hz.delay != dt_filter_1hz.delay
        assert dt_filter_8hz.delay == -0.2455
        assert dt_filter_1hz.delay == -0.1920

    def test_filter_immutability(self, response):
        """Test that filter properties return fresh objects."""
        filter1 = response.magnetic_low_pass
        filter2 = response.magnetic_low_pass

        # Should be equal but not the same object
        # Compare specific attributes instead of full dict due to numpy array comparison issues
        assert filter1.name == filter2.name
        assert filter1.gain == filter2.gain
        assert filter1.normalization_factor == filter2.normalization_factor
        assert filter1 is not filter2

    def test_very_small_dipole_length(self, response):
        """Test dipole filter with very small positive length."""
        filter_obj = response.dipole_filter(1e-10)
        assert filter_obj.gain == 1e-10
        assert "dipole_" in filter_obj.name

    def test_very_large_dipole_length(self, response):
        """Test dipole filter with very large length."""
        filter_obj = response.dipole_filter(1e6)
        assert filter_obj.gain == 1e6
        assert "dipole_" in filter_obj.name


class TestPerformance:
    """Test performance-related aspects."""

    def test_multiple_filter_generation(self, response):
        """Test generating multiple filters efficiently."""
        channels = ["ex", "ey", "hx", "hy", "hz"]

        # Should not raise any errors and complete quickly
        responses = []
        for channel in channels:
            if channel.startswith("e"):
                resp = response.get_channel_response(channel, 100)
            else:
                resp = response.get_channel_response(channel)
            responses.append(resp)

        assert len(responses) == 5
        assert all(isinstance(r, ChannelResponse) for r in responses)

    def test_filter_property_access_performance(self, response):
        """Test that filter properties can be accessed multiple times."""
        # Accessing properties multiple times should work
        for _ in range(10):
            mag_lp = response.magnetic_low_pass
            elec_lp = response.electric_low_pass
            mag_conv = response.magnetic_conversion
            elec_conv = response.electric_conversion

            assert isinstance(mag_lp, PoleZeroFilter)
            assert isinstance(elec_lp, PoleZeroFilter)
            assert isinstance(mag_conv, CoefficientFilter)
            assert isinstance(elec_conv, CoefficientFilter)


# =============================================================================
# Integration Tests
# =============================================================================


class TestResponseIntegration:
    """Test Response integration scenarios."""

    def test_full_electric_workflow(self, response):
        """Test complete electric channel processing workflow."""
        # Set up response for specific configuration
        response.hardware = "PC"
        response.instrument_type = "backbone"
        response.sample_rate = 1

        # Generate complete response
        channel_response = response.get_channel_response("ex", 100.0)

        # Verify complete workflow
        assert isinstance(channel_response, ChannelResponse)
        assert channel_response.units_in == "milliVolt per kilometer"
        assert channel_response.units_out == "digital counts"

        # Verify filter sequence is correct
        filter_names = [f.name for f in channel_response.filters_list]
        expected_sequence = [
            "to_mt_units",
            "dipole_100.00",
            "nims_5_pole_butterworth",
            "nims_1_pole_butterworth",
            "e_analog_to_digital",
            "ex_time_offset",
        ]
        assert filter_names == expected_sequence

    def test_full_magnetic_workflow(self, response):
        """Test complete magnetic channel processing workflow."""
        # Set up response for specific configuration
        response.sample_rate = 8

        # Generate complete response
        channel_response = response.get_channel_response("hx")

        # Verify complete workflow
        assert isinstance(channel_response, ChannelResponse)
        assert channel_response.units_in == "nanoTesla"
        assert channel_response.units_out == "digital counts"

        # Verify filter sequence
        filter_names = [f.name for f in channel_response.filters_list]
        expected_sequence = [
            "nims_3_pole_butterworth",
            "h_analog_to_digital",
            "hx_time_offset",
        ]
        assert filter_names == expected_sequence

    def test_configuration_consistency(self):
        """Test that configuration changes are consistent across filters."""
        response = Response(
            sample_rate=8,
            hardware="HP200",
            h_conversion_factor=200,
            e_conversion_factor=1000000.0,
        )

        # Test magnetic channel
        mag_response = response.get_channel_response("hy")
        mag_conversion = mag_response.filters_list[1]  # Conversion filter
        assert mag_conversion.gain == 200

        # Test electric channel
        elec_response = response.get_channel_response("ey", 50.0)
        elec_conversion = elec_response.filters_list[4]  # Conversion filter
        assert elec_conversion.gain == 1000000.0

        # Test time delays use correct sample rate
        mag_time_delay = mag_response.filters_list[2]
        elec_time_delay = elec_response.filters_list[5]
        assert mag_time_delay.delay == -0.2365  # 8 Hz, hy
        assert elec_time_delay.delay == -0.1525  # 8 Hz, ey


# =============================================================================
# Run Tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
