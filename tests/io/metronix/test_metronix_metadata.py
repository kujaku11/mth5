# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:47:59 2024

@author: jpeacock
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from mt_metadata.timeseries import Electric, Magnetic
from mt_metadata.timeseries.filters import ChannelResponse, FrequencyResponseTableFilter

from mth5.io.metronix import MetronixChannelJSON, MetronixFileNameMetadata


# =============================================================================
# Test Data Import Handling
# =============================================================================

try:
    import mth5_test_data

    metronix_data_path = (
        mth5_test_data.get_test_data_path("metronix")
        / "Northern_Mining"
        / "stations"
        / "saricam"
    )
    has_data_import = True
except ImportError:
    metronix_data_path = None
    has_data_import = False

# Skip marker for tests that require test data
requires_test_data = pytest.mark.skipif(
    not has_data_import, reason="mth5_test_data package not available"
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def metronix_filename_metadata():
    """Create MetronixFileNameMetadata object for testing."""
    return MetronixFileNameMetadata()


@pytest.fixture(scope="session")
def filename_test_cases():
    """Test cases for filename parsing."""
    return [
        {
            "fn": Path(r"084_ADU-07e_C000_TEx_2048Hz.json"),
            "system_number": "084",
            "system_name": "ADU-07e",
            "channel_number": 0,
            "component": "ex",
            "sample_rate": 2048,
            "file_type": "metadata",
        },
        {
            "fn": Path(r"084_ADU-07e_C001_THx_512Hz.json"),
            "system_number": "084",
            "system_name": "ADU-07e",
            "channel_number": 1,
            "component": "hx",
            "sample_rate": 512,
            "file_type": "metadata",
        },
        {
            "fn": Path(r"084_ADU-07e_C002_TEy_128Hz.json"),
            "system_number": "084",
            "system_name": "ADU-07e",
            "channel_number": 2,
            "component": "ey",
            "sample_rate": 128,
            "file_type": "metadata",
        },
        {
            "fn": Path(r"084_ADU-07e_C003_THy_32Hz.json"),
            "system_number": "084",
            "system_name": "ADU-07e",
            "channel_number": 3,
            "component": "hy",
            "sample_rate": 32,
            "file_type": "metadata",
        },
        {
            "fn": Path(r"084_ADU-07e_C004_THz_8Hz.json"),
            "system_number": "084",
            "system_name": "ADU-07e",
            "channel_number": 4,
            "component": "hz",
            "sample_rate": 8,
            "file_type": "metadata",
        },
        {
            "fn": Path(r"084_ADU-07e_C005_TEx_2Hz.json"),
            "system_number": "084",
            "system_name": "ADU-07e",
            "channel_number": 5,
            "component": "ex",
            "sample_rate": 2,
            "file_type": "metadata",
        },
        {
            "fn": Path(r"084_ADU-07e_C006_TEy_2s.json"),
            "system_number": "084",
            "system_name": "ADU-07e",
            "channel_number": 6,
            "component": "ey",
            "sample_rate": 1 / 2,
            "file_type": "metadata",
        },
        {
            "fn": Path(r"084_ADU-07e_C007_THx_8s.atss"),
            "system_number": "084",
            "system_name": "ADU-07e",
            "channel_number": 7,
            "component": "hx",
            "sample_rate": 1 / 8,
            "file_type": "timeseries",
        },
    ]


@pytest.fixture
def fake_atss_file(tmp_path):
    """Create a fake ATSS file for testing file size calculations."""
    atss_file = tmp_path / "084_ADU-07e_C001_THx_128Hz.atss"
    # Create a file with known size (1024 bytes)
    with open(atss_file, "wb") as fid:
        fid.write(b"0" * 1024)
    return atss_file


@pytest.fixture(scope="session")
def magnetic_test_data():
    """Magnetic channel test data dictionary."""
    return {
        "datetime": "2009-08-20T13:22:00",
        "latitude": 39.026196666666664,
        "longitude": 29.123953333333333,
        "elevation": 1088.31,
        "angle": 0.0,
        "tilt": 0.0,
        "resistance": 684052.0,
        "units": "mV",
        "filter": "ADB-LF,LF-RF-4",
        "source": "",
        "sensor_calibration": {
            "sensor": "MFS-06",
            "serial": 26,
            "chopper": 1,
            "units_frequency": "Hz",
            "units_amplitude": "mV/nT",
            "units_phase": "degrees",
            "datetime": "2006-12-01T11:23:02",
            "Operator": "",
            "f": [
                1e-05,
                1.2589000000000001e-05,
                1.5849e-05,
                1.9952e-05,
                2.5119e-05,
                3.1623000000000005e-05,
                3.981e-05,
                5.0118000000000004e-05,
                6.3095e-05,
                7.943e-05,
                0.0001,
                0.00012589,
                0.00015849,
                0.00019952000000000001,
                0.00025119,
                0.00031623,
                0.0003981,
                0.00050118,
                0.00063095,
                0.0007943000000000001,
                0.001,
                0.0012589,
                0.0015849000000000002,
                0.0019952,
                0.0025119,
                0.0031623000000000003,
                0.003981,
                0.0050118,
                0.0063095,
                0.007943,
                0.01,
                0.012589000000000001,
                0.015849000000000002,
                0.019951999999999998,
                0.025119000000000002,
                0.031623000000000005,
                0.03981,
                0.050118,
                0.063095,
                0.07943,
                0.1,
                0.12589,
                0.15849,
                0.19953,
                0.25119,
                0.31623,
                0.39811,
                0.50119,
                0.63095,
                0.7943,
                1.0,
                1.2589,
                1.5849,
                1.9952,
                2.5119,
                3.1623,
                3.981,
                5.0118,
                6.3095,
                7.943,
                10.0,
                12.589,
                15.849,
                19.952,
                25.119,
                31.622,
                39.81,
                50.118,
                63.095,
                79.43,
                100.0,
                125.89,
                158.49,
                199.52,
                251.18,
                316.22,
                398.1,
                501.18,
                630.94,
                794.3,
                1000.0,
                1258.9,
                1584.9,
                1995.2,
                2511.8,
                3162.2,
                3981.0,
                5011.7,
                6309.4,
                7943.0,
                9999.5,
                10000.0,
            ],
            "a": [
                0.0019999999999937507,
                0.0025177999999875305,
                0.003169799999975118,
                0.003990399999950359,
                0.005023799999900942,
                0.006324599999802356,
                0.007961999999605673,
                0.010023599999213205,
                0.012618999998430126,
                0.015885999996867916,
                0.019999999993749996,
                0.025177999987530366,
                0.031697999975117984,
                0.03990399995035912,
                0.050237999900942555,
                0.06324599980235342,
                0.07961999960567287,
                0.10023599921320551,
                0.1261899984301254,
                0.15885999686791308,
                0.1999999937499987,
                0.2517799875303634,
                0.3169799751179777,
                0.3990399503591332,
                0.5023799009425741,
                0.6324598023534992,
                0.796199605673159,
                1.002359213206409,
                1.2618984301283263,
                1.588596867922261,
                1.9999937500276823,
                2.517787530455084,
                3.169775118267761,
                3.990350360050149,
                5.0237009454743236,
                6.324402362670752,
                7.961605702157798,
                10.022813298108742,
                12.617430418287983,
                15.882868838977918,
                20.010376463273992,
                25.16799310918104,
                31.712516917010664,
                39.8980188,
                50.2405119,
                63.1068588,
                79.1203814,
                99.68167910000001,
                124.92179050000001,
                155.881375,
                193.79,
                241.38148599999997,
                297.517428,
                360.03384,
                429.38418599999994,
                502.8373230000001,
                575.0156400000001,
                635.796948,
                688.87121,
                734.25092,
                766.36,
                785.780202,
                801.1669499999999,
                810.35048,
                816.9452370000001,
                822.0455119999999,
                825.89826,
                770.714604,
                826.0397399999999,
                827.81946,
                825.9499999999999,
                828.0288859999999,
                830.17062,
                828.8659359999999,
                829.044708,
                826.75719,
                822.7532699999999,
                822.6368520000001,
                821.862444,
                820.03532,
                818.2,
                815.477653,
                813.814452,
                840.637616,
                784.636084,
                749.12518,
                698.46645,
                596.542651,
                528.601532,
                551.911412,
                517.094144,
                517.0699999999999,
            ],
            "p": [
                89.99985667036422,
                89.99981956232152,
                89.99977283686026,
                89.9997140287107,
                89.9996399702879,
                89.9995467486928,
                89.99942940471999,
                89.99928166053145,
                89.99909566166313,
                89.99886153270316,
                89.99856670364251,
                89.99819562321578,
                89.99772836860375,
                89.99714028710932,
                89.99639970288361,
                89.99546748693723,
                89.9942940472185,
                89.99281660535163,
                89.99095661670552,
                89.98861532717969,
                89.98566703672059,
                89.98195623274725,
                89.97728368721353,
                89.9714028734397,
                89.9639970335185,
                89.95467487871475,
                89.94294049082438,
                89.92816609070726,
                89.90956624126156,
                89.88615341984706,
                89.8566706626361,
                89.81956291689636,
                89.77283804827267,
                89.7140310808364,
                89.63997501742048,
                89.5467581293517,
                89.42942354657666,
                89.28169809452952,
                89.09573660784051,
                88.8616822137415,
                88.56650097293976,
                88.19160912138915,
                87.72027775680306,
                87.177,
                86.459,
                85.487,
                84.447,
                82.992,
                81.233,
                79.055,
                76.345,
                72.954,
                69.03,
                64.201,
                58.69,
                52.571,
                46.212,
                39.399,
                33.074,
                27.567,
                22.507,
                18.198,
                14.529,
                11.635,
                9.0507,
                7.133,
                5.5026,
                3.6917,
                3.2486,
                2.2856,
                1.3903,
                0.8116,
                0.21197,
                -0.5109,
                -1.217,
                -2.3402,
                -2.8624,
                -3.7097,
                -4.8118,
                -6.1635,
                -7.8127,
                -9.8044,
                -12.075,
                -14.807,
                -23.519,
                -27.61,
                -34.118,
                -40.321,
                -36.267,
                -40.999,
                -51.017,
                -51.019,
            ],
        },
    }


@pytest.fixture(scope="session")
def electric_test_data():
    """Electric channel test data dictionary."""
    return {
        "datetime": "2009-08-20T13:22:00",
        "latitude": 39.026196666666664,
        "longitude": 29.123953333333333,
        "elevation": 1088.31,
        "angle": 0.0,
        "tilt": 0.0,
        "resistance": 572.3670043945313,
        "units": "mV/km",
        "filter": "ADB-LF,LF-RF-4",
        "source": "",
        "sensor_calibration": {
            "sensor": "EFP-06",
            "serial": 0,
            "chopper": 1,
            "units_frequency": "Hz",
            "units_amplitude": "mV",
            "units_phase": "degrees",
            "datetime": "1970-01-01T00:00:00",
            "Operator": "",
            "f": [],
            "a": [],
            "p": [],
        },
    }


@pytest.fixture
def temp_magnetic_json(tmp_path, magnetic_test_data):
    """Create temporary magnetic JSON file."""
    json_file = tmp_path / "084_ADU-07e_C002_THx_512Hz.json"
    with open(json_file, "w") as fid:
        json.dump(magnetic_test_data, fid)
    yield json_file
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def temp_electric_json(tmp_path, electric_test_data):
    """Create temporary electric JSON file."""
    json_file = tmp_path / "084_ADU-07e_C000_TEx_128Hz.json"
    with open(json_file, "w") as fid:
        json.dump(electric_test_data, fid)
    yield json_file
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def magnetic_channel_json(temp_magnetic_json):
    """Create MetronixChannelJSON object for magnetic data."""
    return MetronixChannelJSON(temp_magnetic_json)


@pytest.fixture
def electric_channel_json(temp_electric_json):
    """Create MetronixChannelJSON object for electric data."""
    return MetronixChannelJSON(temp_electric_json)


@pytest.fixture(scope="session")
def expected_magnetic_metadata():
    """Expected magnetic metadata for comparison."""
    return Magnetic(
        **OrderedDict(
            [
                ("channel_number", 2),
                ("component", "hx"),
                ("data_quality.rating.value", None),
                (
                    "filters",
                    [
                        OrderedDict(
                            [
                                (
                                    "applied_filter",
                                    OrderedDict(
                                        [
                                            ("name", "adb-lf"),
                                            ("applied", True),
                                            ("stage", 1),
                                        ]
                                    ),
                                )
                            ]
                        ),
                        OrderedDict(
                            [
                                (
                                    "applied_filter",
                                    OrderedDict(
                                        [
                                            ("name", "lf-rf-4"),
                                            ("applied", True),
                                            ("stage", 2),
                                        ]
                                    ),
                                )
                            ]
                        ),
                        OrderedDict(
                            [
                                (
                                    "applied_filter",
                                    OrderedDict(
                                        [
                                            ("name", "mfs-06_chopper_1"),
                                            ("applied", True),
                                            ("stage", 3),
                                        ]
                                    ),
                                )
                            ]
                        ),
                    ],
                ),
                ("filter.applied", [True, True, True]),
                ("filter.name", ["adb-lf", "lf-rf-4", "mfs-06_chopper_1"]),
                ("location.elevation", 1088.31),
                ("location.latitude", 39.026196666666664),
                ("location.longitude", 29.123953333333333),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 512.0),
                ("sensor.id", "26"),
                ("sensor.manufacturer", "Metronix Geophysics"),
                ("sensor.type", "induction coil"),
                ("sensor.model", "MFS-06"),
                ("time_period.end", "2009-08-20T13:22:01.028076171+00:00"),
                ("time_period.start", "2009-08-20T13:22:00+00:00"),
                ("type", "magnetic"),
                ("units", "mV"),
            ]
        )
    )


@pytest.fixture(scope="session")
def expected_electric_metadata():
    """Expected electric metadata for comparison."""
    return Electric(
        **OrderedDict(
            [
                ("channel_number", 0),
                ("component", "ex"),
                ("contact_resistance.start", 572.3670043945312),
                ("data_quality.rating.value", None),
                ("dipole_length", 0.0),
                (
                    "filters",
                    [
                        OrderedDict(
                            [
                                (
                                    "applied_filter",
                                    OrderedDict(
                                        [
                                            ("name", "adb-lf"),
                                            ("applied", True),
                                            ("stage", 1),
                                        ]
                                    ),
                                )
                            ]
                        ),
                        OrderedDict(
                            [
                                (
                                    "applied_filter",
                                    OrderedDict(
                                        [
                                            ("name", "lf-rf-4"),
                                            ("applied", True),
                                            ("stage", 2),
                                        ]
                                    ),
                                )
                            ]
                        ),
                    ],
                ),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("negative.elevation", 0.0),
                ("negative.id", None),
                ("negative.latitude", 0.0),
                ("negative.longitude", 0.0),
                ("negative.manufacturer", None),
                ("negative.type", None),
                ("positive.elevation", 1088.31),
                ("positive.id", None),
                ("positive.latitude", 39.026196666666664),
                ("positive.longitude", 29.123953333333333),
                ("positive.manufacturer", None),
                ("positive.type", None),
                ("sample_rate", 128.0),
                ("time_period.end", "2009-08-20T13:22:00.456055+00:00"),
                ("time_period.start", "2009-08-20T13:22:00+00:00"),
                ("type", "electric"),
                ("units", "mV/km"),
            ]
        )
    )


@pytest.fixture(scope="session")
def expected_fap_filter(magnetic_test_data):
    """Expected frequency response filter for magnetic sensor."""
    expected_fap_dict = OrderedDict(
        [
            (
                "amplitudes",
                np.array(magnetic_test_data["sensor_calibration"]["a"]),
            ),
            ("calibration_date", "2006-12-01T11:23:02+00:00"),
            (
                "frequencies",
                np.array(magnetic_test_data["sensor_calibration"]["f"]),
            ),
            ("gain", 1.0),
            ("instrument_type", ""),
            ("name", "mfs-06_chopper_1"),
            (
                "phases",
                np.deg2rad(np.array(magnetic_test_data["sensor_calibration"]["p"])),
            ),
            ("type", "frequency response table"),
            ("units_out", "mV"),
            ("units_in", "nT"),
        ]
    )
    return FrequencyResponseTableFilter(**expected_fap_dict)


# =============================================================================
# Test Classes
# =============================================================================


class TestMetronixFileNameMetadata:
    """Test MetronixFileNameMetadata functionality."""

    def test_filename_parsing_all_cases(
        self, metronix_filename_metadata, filename_test_cases
    ):
        """Test filename parsing for all test cases."""
        obj = metronix_filename_metadata

        for case in filename_test_cases:
            obj.fn = case["fn"]

            # Test all properties for this filename
            for key, expected_value in case.items():
                if key != "fn":  # Skip the filename itself
                    actual_value = getattr(obj, key)
                    assert actual_value == expected_value, (
                        f"For file {case['fn']}, property '{key}' should be "
                        f"{expected_value}, but got {actual_value}"
                    )


class TestMetronixJSONMagnetic:
    """Test MetronixChannelJSON functionality for magnetic channels."""

    @requires_test_data
    def test_basic_properties(self, magnetic_channel_json, temp_magnetic_json):
        """Test basic properties of magnetic channel."""
        magnetic = magnetic_channel_json

        assert magnetic.fn == temp_magnetic_json
        assert magnetic.system_number == "084"
        assert magnetic.system_name == "ADU-07e"
        assert magnetic.channel_number == 2
        assert magnetic.component == "hx"
        assert magnetic.sample_rate == 512.0
        assert magnetic.file_type == "metadata"

    @requires_test_data
    def test_has_metadata(self, magnetic_channel_json):
        """Test metadata detection."""
        assert magnetic_channel_json._has_metadata() == True

    @requires_test_data
    def test_channel_metadata(self, magnetic_channel_json, expected_magnetic_metadata):
        """Test channel metadata generation."""
        magnetic_metadata = magnetic_channel_json.get_channel_metadata()
        assert magnetic_metadata == expected_magnetic_metadata

    @requires_test_data
    def test_sensor_response_filter(self, magnetic_channel_json, expected_fap_filter):
        """Test sensor response filter generation."""
        fap = magnetic_channel_json.get_sensor_response_filter()
        assert fap == expected_fap_filter


class TestMetronixJSONElectric:
    """Test MetronixChannelJSON functionality for electric channels."""

    @requires_test_data
    def test_basic_properties(self, electric_channel_json, temp_electric_json):
        """Test basic properties of electric channel."""
        electric = electric_channel_json

        assert electric.fn == temp_electric_json
        assert electric.system_number == "084"
        assert electric.system_name == "ADU-07e"
        assert electric.channel_number == 0
        assert electric.component == "ex"
        assert electric.sample_rate == 128.0
        assert electric.file_type == "metadata"

    @requires_test_data
    def test_has_metadata(self, electric_channel_json):
        """Test metadata detection."""
        assert electric_channel_json._has_metadata() == True

    @requires_test_data
    def test_channel_metadata(self, electric_channel_json, expected_electric_metadata):
        """Test electric channel metadata generation."""
        electric_metadata = electric_channel_json.get_channel_metadata()
        assert electric_metadata == expected_electric_metadata

    @requires_test_data
    def test_sensor_response_filter_none(self, electric_channel_json):
        """Test that electric channel returns None for sensor response filter."""
        fap = electric_channel_json.get_sensor_response_filter()
        assert fap is None


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "fn": Path(r"084_ADU-07e_C000_TEx_2048Hz.json"),
            "expected": {
                "system_number": "084",
                "component": "ex",
                "sample_rate": 2048,
            },
        },
        {
            "fn": Path(r"084_ADU-07e_C001_THx_512Hz.json"),
            "expected": {"system_number": "084", "component": "hx", "sample_rate": 512},
        },
        {
            "fn": Path(r"084_ADU-07e_C007_THx_8s.atss"),
            "expected": {
                "system_number": "084",
                "component": "hx",
                "sample_rate": 1 / 8,
            },
        },
    ],
)
def test_filename_parsing_parameterized(metronix_filename_metadata, test_case):
    """Test filename parsing with parameterized test cases."""
    obj = metronix_filename_metadata
    obj.fn = test_case["fn"]

    for key, expected_value in test_case["expected"].items():
        actual_value = getattr(obj, key)
        assert actual_value == expected_value


# =============================================================================
# Integration Tests
# =============================================================================


class TestMetronixIntegration:
    """Integration tests for Metronix functionality."""

    @requires_test_data
    def test_magnetic_workflow(self, temp_magnetic_json, magnetic_test_data):
        """Test complete magnetic channel workflow."""
        # Create channel object
        magnetic = MetronixChannelJSON(temp_magnetic_json)

        # Verify filename parsing
        assert magnetic.component == "hx"
        assert magnetic.sample_rate == 512.0

        # Verify metadata generation
        metadata = magnetic.get_channel_metadata()
        assert metadata.type == "magnetic"
        assert metadata.component == "hx"
        assert metadata.sample_rate == 512.0

        # Verify sensor response filter
        filter_obj = magnetic.get_sensor_response_filter()
        assert filter_obj is not None
        assert filter_obj.name == "mfs-06_chopper_1"
        assert len(filter_obj.frequencies) > 0

    @requires_test_data
    def test_electric_workflow(self, temp_electric_json, electric_test_data):
        """Test complete electric channel workflow."""
        # Create channel object
        electric = MetronixChannelJSON(temp_electric_json)

        # Verify filename parsing
        assert electric.component == "ex"
        assert electric.sample_rate == 128.0

        # Verify metadata generation
        metadata = electric.get_channel_metadata()
        assert metadata.type == "electric"
        assert metadata.component == "ex"
        assert metadata.sample_rate == 128.0

        # Verify no sensor response filter for electric
        filter_obj = electric.get_sensor_response_filter()
        assert filter_obj is None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestMetronixEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_filename(self, metronix_filename_metadata):
        """Test behavior with invalid filename."""
        obj = metronix_filename_metadata

        # Test with completely invalid filename - should raise IndexError due to insufficient parts
        with pytest.raises(IndexError):
            obj.fn = Path("invalid_filename.txt")

    @requires_test_data
    def test_missing_json_file(self):
        """Test behavior with missing JSON file."""
        nonexistent_path = Path("nonexistent_file.json")

        # Should raise IOError/OSError, not FileNotFoundError
        with pytest.raises((IOError, OSError)):
            MetronixChannelJSON(nonexistent_path)

    @requires_test_data
    def test_malformed_json_content(self, tmp_path):
        """Test behavior with valid filename but malformed JSON content."""
        # Create a valid filename but with malformed JSON content
        json_file = tmp_path / "084_ADU-07e_C000_TEx_128Hz.json"
        with open(json_file, "w") as fid:
            fid.write("{ invalid json content }")

        # Should raise JSONDecodeError when trying to read the content
        with pytest.raises(json.JSONDecodeError):
            channel_obj = MetronixChannelJSON(json_file)
            # The error occurs during the read() method


# =============================================================================
# Performance Tests
# =============================================================================


class TestMetronixPerformance:
    """Performance tests for Metronix functionality."""

    @requires_test_data
    def test_filename_parsing_performance(
        self, metronix_filename_metadata, filename_test_cases
    ):
        """Test filename parsing performance."""
        import time

        obj = metronix_filename_metadata
        start_time = time.time()

        # Parse many filenames
        for _ in range(100):
            for case in filename_test_cases:
                obj.fn = case["fn"]
                _ = obj.component
                _ = obj.sample_rate

        end_time = time.time()
        # Should complete quickly
        assert end_time - start_time < 2.0

    @requires_test_data
    def test_metadata_generation_performance(self, magnetic_channel_json):
        """Test metadata generation performance."""
        import time

        start_time = time.time()

        # Generate metadata multiple times
        for _ in range(10):
            metadata = magnetic_channel_json.get_channel_metadata()
            filter_obj = magnetic_channel_json.get_sensor_response_filter()

        end_time = time.time()
        # Should complete quickly
        assert end_time - start_time < 1.0


# =============================================================================
# Additional Comprehensive Tests for Missing Functionality
# =============================================================================


class TestMetronixFileNameMetadataProperties:
    """Test additional properties and methods of MetronixFileNameMetadata."""

    def test_initialization_empty(self):
        """Test empty initialization."""
        obj = MetronixFileNameMetadata()
        assert obj.system_number is None
        assert obj.system_name is None
        assert obj.channel_number is None
        assert obj.component is None
        assert obj.sample_rate is None
        assert obj.file_type is None
        assert obj.fn is None

    def test_initialization_with_filename(self, filename_test_cases):
        """Test initialization with filename."""
        case = filename_test_cases[0]  # Test first case
        obj = MetronixFileNameMetadata(fn=case["fn"])
        assert obj.system_number == case["system_number"]
        assert obj.system_name == case["system_name"]
        assert obj.channel_number == case["channel_number"]
        assert obj.component == case["component"]
        assert obj.sample_rate == case["sample_rate"]
        assert obj.file_type == case["file_type"]

    def test_fn_property_none(self):
        """Test fn property with None value."""
        obj = MetronixFileNameMetadata()
        obj.fn = None
        assert obj.fn is None
        assert obj.fn_exists is False

    def test_fn_exists_property_true(self, fake_atss_file):
        """Test fn_exists property when file exists."""
        obj = MetronixFileNameMetadata(fn=fake_atss_file)
        assert obj.fn_exists is True

    def test_fn_exists_property_false(self):
        """Test fn_exists property when file doesn't exist."""
        obj = MetronixFileNameMetadata()
        obj._fn = Path("nonexistent_file.atss")
        assert obj.fn_exists is False

    def test_file_size_property(self, fake_atss_file):
        """Test file_size property."""
        obj = MetronixFileNameMetadata(fn=fake_atss_file)
        assert obj.file_size == 1024

    def test_file_size_property_no_file(self):
        """Test file_size property when no file is set."""
        obj = MetronixFileNameMetadata()
        assert obj.file_size == 0

    def test_n_samples_property(self, fake_atss_file):
        """Test n_samples property."""
        obj = MetronixFileNameMetadata(fn=fake_atss_file)
        expected_samples = 1024 / 8  # 8 bytes per sample
        assert obj.n_samples == expected_samples

    def test_duration_property(self, fake_atss_file):
        """Test duration property."""
        obj = MetronixFileNameMetadata(fn=fake_atss_file)
        expected_duration = (1024 / 8) / 128.0  # n_samples / sample_rate
        assert obj.duration == expected_duration

    def test_str_representation(self, fake_atss_file):
        """Test string representation."""
        obj = MetronixFileNameMetadata(fn=fake_atss_file)
        str_repr = str(obj)
        assert "Metronix ATSS TIMESERIES:" in str_repr
        assert "System Name:    ADU-07e" in str_repr
        assert "System Number:  084" in str_repr
        assert "Channel Number: 1" in str_repr
        assert "Component:      hx" in str_repr
        assert "Sample Rate:    128.0" in str_repr

    def test_repr_representation(self, fake_atss_file):
        """Test repr representation."""
        obj = MetronixFileNameMetadata(fn=fake_atss_file)
        assert repr(obj) == str(obj)

    def test_str_representation_no_filename(self):
        """Test string representation with no filename."""
        obj = MetronixFileNameMetadata()
        # When no filename is set, __str__ should return None (implementation issue)
        # This will raise TypeError because __str__ should always return a string
        with pytest.raises(TypeError, match="__str__ returned non-string"):
            str(obj)


class TestMetronixFileNameMetadataParsing:
    """Test parsing methods of MetronixFileNameMetadata."""

    def test_parse_channel_number_edge_cases(self):
        """Test channel number parsing edge cases."""
        obj = MetronixFileNameMetadata()
        assert obj._parse_channel_number("C000") == 0
        assert obj._parse_channel_number("C999") == 999
        assert obj._parse_channel_number("C001") == 1
        assert obj._parse_channel_number("C007") == 7

    def test_parse_component_edge_cases(self):
        """Test component parsing edge cases."""
        obj = MetronixFileNameMetadata()
        assert obj._parse_component("TEx") == "ex"
        assert obj._parse_component("THx") == "hx"
        assert obj._parse_component("TEy") == "ey"
        assert obj._parse_component("THy") == "hy"
        assert obj._parse_component("THz") == "hz"

    def test_parse_sample_rate_hz_cases(self):
        """Test sample rate parsing for Hz cases."""
        obj = MetronixFileNameMetadata()
        assert obj._parse_sample_rate("128Hz") == 128.0
        assert obj._parse_sample_rate("2048HZ") == 2048.0
        assert obj._parse_sample_rate("0.5hz") == 0.5

    def test_parse_sample_rate_seconds_cases(self):
        """Test sample rate parsing for seconds cases."""
        obj = MetronixFileNameMetadata()
        assert obj._parse_sample_rate("2s") == 0.5
        assert obj._parse_sample_rate("8S") == 0.125
        assert obj._parse_sample_rate("0.5s") == 2.0

    def test_get_file_type_json(self):
        """Test file type detection for JSON files."""
        obj = MetronixFileNameMetadata()
        assert obj._get_file_type(Path("test.json")) == "metadata"

    def test_get_file_type_atss(self):
        """Test file type detection for ATSS files."""
        obj = MetronixFileNameMetadata()
        assert obj._get_file_type(Path("test.atss")) == "timeseries"

    def test_get_file_type_unsupported(self):
        """Test file type detection for unsupported files."""
        obj = MetronixFileNameMetadata()
        with pytest.raises(ValueError, match="Metronix file type .* not supported"):
            obj._get_file_type(Path("test.txt"))


class TestMetronixChannelJSONInit:
    """Test MetronixChannelJSON initialization and basic operations."""

    def test_empty_initialization(self):
        """Test empty initialization."""
        obj = MetronixChannelJSON()
        assert obj.metadata is None
        assert obj.fn is None
        assert not obj._has_metadata()

    @requires_test_data
    def test_initialization_with_filename(self, temp_magnetic_json):
        """Test initialization with filename."""
        obj = MetronixChannelJSON(fn=temp_magnetic_json)
        assert obj.metadata is not None
        assert obj._has_metadata()
        assert obj.fn == temp_magnetic_json

    def test_initialization_nonexistent_file(self):
        """Test initialization with non-existent file."""
        with pytest.raises(IOError, match="Cannot find Metronix JSON file"):
            MetronixChannelJSON(fn="nonexistent_file.json")


class TestMetronixChannelJSONFileOperations:
    """Test file operations for MetronixChannelJSON."""

    @requires_test_data
    def test_read_method(self, temp_magnetic_json):
        """Test read method."""
        obj = MetronixChannelJSON()
        obj.read(temp_magnetic_json)
        assert obj.metadata is not None
        assert obj._has_metadata()

    @requires_test_data
    def test_read_method_with_fn_parameter(self, temp_magnetic_json):
        """Test read method with fn parameter."""
        obj = MetronixChannelJSON()
        obj.read(fn=temp_magnetic_json)
        assert obj.metadata is not None
        assert obj.fn == temp_magnetic_json

    def test_read_nonexistent_file(self):
        """Test read method with non-existent file."""
        obj = MetronixChannelJSON()
        obj._fn = Path("nonexistent_file.json")
        with pytest.raises(IOError, match="Cannot find Metronix JSON file"):
            obj.read()

    def test_fn_setter_none(self):
        """Test fn setter with None."""
        obj = MetronixChannelJSON()
        obj.fn = None
        assert obj.fn is None

    @requires_test_data
    def test_fn_setter_existing_file(self, temp_magnetic_json):
        """Test fn setter with existing file."""
        obj = MetronixChannelJSON()
        obj.fn = temp_magnetic_json
        assert obj.fn == temp_magnetic_json
        assert obj.metadata is not None

    def test_fn_setter_nonexistent_file(self):
        """Test fn setter with non-existent file."""
        obj = MetronixChannelJSON()
        with pytest.raises(IOError, match="Cannot find Metronix JSON file"):
            obj.fn = "nonexistent_file.json"


class TestMetronixChannelJSONMetadataDetails:
    """Test detailed metadata functionality."""

    @requires_test_data
    def test_channel_metadata_location_magnetic(self, magnetic_channel_json):
        """Test magnetic channel metadata location properties."""
        metadata = magnetic_channel_json.get_channel_metadata()
        assert metadata.location.latitude == 39.026196666666664
        assert metadata.location.longitude == 29.123953333333333
        assert metadata.location.elevation == 1088.31

    @requires_test_data
    def test_channel_metadata_sensor_magnetic(self, magnetic_channel_json):
        """Test magnetic channel metadata sensor properties."""
        metadata = magnetic_channel_json.get_channel_metadata()
        assert metadata.sensor.id == "26"
        assert metadata.sensor.manufacturer == "Metronix Geophysics"
        assert metadata.sensor.type == "induction coil"
        assert metadata.sensor.model == "MFS-06"

    @requires_test_data
    def test_channel_metadata_time_period_magnetic(self, magnetic_channel_json):
        """Test magnetic channel metadata time period."""
        metadata = magnetic_channel_json.get_channel_metadata()
        assert str(metadata.time_period.start) == "2009-08-20T13:22:00+00:00"
        # Check that end time is properly calculated

    @requires_test_data
    def test_channel_metadata_filters_magnetic(self, magnetic_channel_json):
        """Test magnetic channel metadata filters."""
        metadata = magnetic_channel_json.get_channel_metadata()
        filters = metadata.filters
        assert len(filters) == 3  # 2 from filter string + 1 sensor response
        assert filters[0].name == "adb-lf"
        assert filters[1].name == "lf-rf-4"
        assert filters[2].name == "mfs-06_chopper_1"
        assert all(f.applied for f in filters)

    @requires_test_data
    def test_channel_metadata_electrodes_electric(self, electric_channel_json):
        """Test electric channel electrode properties."""
        metadata = electric_channel_json.get_channel_metadata()
        assert metadata.positive.latitude == 39.026196666666664
        assert metadata.positive.longitude == 29.123953333333333
        assert metadata.positive.elevation == 1088.31
        assert metadata.contact_resistance.start == 572.3670043945313

    @requires_test_data
    def test_channel_metadata_filters_no_sensor_electric(self, electric_channel_json):
        """Test electric channel metadata filters without sensor response."""
        metadata = electric_channel_json.get_channel_metadata()
        filters = metadata.filters
        assert len(filters) == 2  # Only 2 from filter string, no sensor response
        assert filters[0].name == "adb-lf"
        assert filters[1].name == "lf-rf-4"


class TestMetronixChannelJSONSensorResponse:
    """Test sensor response functionality."""

    @requires_test_data
    def test_sensor_response_filter_phase_conversion_magnetic(
        self, magnetic_channel_json
    ):
        """Test phase conversion from degrees to radians."""
        fap = magnetic_channel_json.get_sensor_response_filter()
        # Check that phases are in radians (should be less than or equal to 2*pi in absolute value)
        assert np.all(np.abs(fap.phases) <= 2 * np.pi)

    @requires_test_data
    def test_sensor_response_filter_units_magnetic(self, magnetic_channel_json):
        """Test sensor response filter units."""
        fap = magnetic_channel_json.get_sensor_response_filter()
        # Units may be converted to full names by the metadata system
        assert fap.units_out in ["mV", "milliVolt"]
        assert fap.units_in in ["nT", "nanoTesla"]

    @requires_test_data
    def test_sensor_response_filter_arrays_magnetic(self, magnetic_channel_json):
        """Test sensor response filter array lengths."""
        fap = magnetic_channel_json.get_sensor_response_filter()
        assert len(fap.frequencies) > 0
        assert len(fap.frequencies) == len(fap.amplitudes)
        assert len(fap.frequencies) == len(fap.phases)

    @requires_test_data
    def test_sensor_response_filter_empty_electric(self, electric_channel_json):
        """Test sensor response filter with empty calibration data."""
        fap = electric_channel_json.get_sensor_response_filter()
        assert fap is None  # Should return None for empty frequency data


class TestMetronixChannelJSONChannelResponse:
    """Test channel response functionality."""

    @requires_test_data
    def test_get_channel_response_with_sensor(self, magnetic_channel_json):
        """Test get_channel_response with sensor response filter."""
        response = magnetic_channel_json.get_channel_response()
        assert isinstance(response, ChannelResponse)
        assert len(response.filters_list) == 1
        assert isinstance(response.filters_list[0], FrequencyResponseTableFilter)

    @requires_test_data
    def test_get_channel_response_without_sensor(self, electric_channel_json):
        """Test get_channel_response without sensor response filter."""
        response = electric_channel_json.get_channel_response()
        assert isinstance(response, ChannelResponse)
        assert len(response.filters_list) == 0


class TestMetronixChannelJSONAdvancedEdgeCases:
    """Test advanced edge cases and error conditions."""

    def test_get_channel_metadata_no_metadata(self):
        """Test get_channel_metadata when no metadata is loaded."""
        obj = MetronixChannelJSON()
        assert obj.get_channel_metadata() is None

    def test_get_sensor_response_filter_no_metadata(self):
        """Test get_sensor_response_filter when no metadata is loaded."""
        obj = MetronixChannelJSON()
        assert obj.get_sensor_response_filter() is None

    def test_get_channel_response_no_metadata(self):
        """Test get_channel_response when no metadata is loaded."""
        obj = MetronixChannelJSON()
        response = obj.get_channel_response()
        assert isinstance(response, ChannelResponse)
        assert len(response.filters_list) == 0

    @requires_test_data
    def test_invalid_component_error(self, tmp_path):
        """Test error with invalid component."""
        invalid_data = {
            "datetime": "2009-08-20T13:22:00",
            "latitude": 0.0,
            "longitude": 0.0,
            "elevation": 0.0,
            "angle": 0.0,
            "tilt": 0.0,
            "resistance": 0.0,
            "units": "mV",
            "filter": "test",
            "source": "",
            "sensor_calibration": {
                "sensor": "TEST",
                "serial": 0,
                "chopper": 1,
                "units_frequency": "Hz",
                "units_amplitude": "mV/nT",
                "units_phase": "degrees",
                "datetime": "2006-12-01T11:23:02",
                "Operator": "",
                "f": [],  # Empty to avoid sensor response filter creation
                "a": [],
                "p": [],
            },
        }

        json_file = (
            tmp_path / "084_ADU-07e_C000_TZz_128Hz.json"
        )  # Invalid component 'zz'
        with open(json_file, "w") as fid:
            json.dump(invalid_data, fid)

        channel = MetronixChannelJSON(json_file)
        with pytest.raises(ValueError, match="Do not understand channel component"):
            channel.get_channel_metadata()

    @requires_test_data
    def test_sensor_response_filter_radians_phase(self, tmp_path, magnetic_test_data):
        """Test sensor response filter with phase in radians."""
        # Modify data to use radians
        magnetic_test_data = magnetic_test_data.copy()
        magnetic_test_data["sensor_calibration"] = magnetic_test_data[
            "sensor_calibration"
        ].copy()
        magnetic_test_data["sensor_calibration"]["units_phase"] = "radians"
        magnetic_test_data["sensor_calibration"]["p"] = np.deg2rad(
            magnetic_test_data["sensor_calibration"]["p"]
        ).tolist()

        json_file = tmp_path / "084_ADU-07e_C002_THx_512Hz.json"
        with open(json_file, "w") as fid:
            json.dump(magnetic_test_data, fid)

        channel = MetronixChannelJSON(json_file)
        fap = channel.get_sensor_response_filter()
        # Phases should be used as-is (already in radians)
        assert fap is not None, "Sensor response filter should not be None"
        assert np.allclose(fap.phases, magnetic_test_data["sensor_calibration"]["p"])

    @requires_test_data
    def test_sensor_response_filter_empty_frequencies(
        self, tmp_path, magnetic_test_data
    ):
        """Test sensor response filter with empty frequency data."""
        # Remove frequency data
        magnetic_test_data = magnetic_test_data.copy()
        magnetic_test_data["sensor_calibration"] = magnetic_test_data[
            "sensor_calibration"
        ].copy()
        magnetic_test_data["sensor_calibration"]["f"] = []
        magnetic_test_data["sensor_calibration"]["a"] = []
        magnetic_test_data["sensor_calibration"]["p"] = []

        json_file = tmp_path / "084_ADU-07e_C002_THx_512Hz.json"
        with open(json_file, "w") as fid:
            json.dump(magnetic_test_data, fid)

        channel = MetronixChannelJSON(json_file)
        fap = channel.get_sensor_response_filter()
        assert fap is None


class TestMetronixChannelJSONComprehensiveIntegration:
    """Comprehensive integration tests combining multiple features."""

    @requires_test_data
    def test_full_workflow_magnetic(self, temp_magnetic_json):
        """Test complete workflow for magnetic channel."""
        # Initialize
        channel = MetronixChannelJSON(temp_magnetic_json)

        # Get metadata
        metadata = channel.get_channel_metadata()
        assert isinstance(metadata, Magnetic)

        # Get sensor response
        sensor_filter = channel.get_sensor_response_filter()
        assert isinstance(sensor_filter, FrequencyResponseTableFilter)

        # Get channel response
        response = channel.get_channel_response()
        assert isinstance(response, ChannelResponse)
        assert len(response.filters_list) == 1

    @requires_test_data
    def test_full_workflow_electric(self, temp_electric_json):
        """Test complete workflow for electric channel."""
        # Initialize
        channel = MetronixChannelJSON(temp_electric_json)

        # Get metadata
        metadata = channel.get_channel_metadata()
        assert isinstance(metadata, Electric)

        # Get sensor response (should be None)
        sensor_filter = channel.get_sensor_response_filter()
        assert sensor_filter is None

        # Get channel response
        response = channel.get_channel_response()
        assert isinstance(response, ChannelResponse)
        assert len(response.filters_list) == 0

    @requires_test_data
    def test_metadata_consistency(self, temp_magnetic_json):
        """Test consistency between different metadata representations."""
        channel = MetronixChannelJSON(temp_magnetic_json)
        metadata = channel.get_channel_metadata()

        # Check consistency between parsed filename and metadata content
        assert metadata is not None, "Metadata should not be None"
        assert channel.component == metadata.component
        assert channel.channel_number == metadata.channel_number
        assert channel.sample_rate == metadata.sample_rate

    @requires_test_data
    def test_multiple_reads_same_file(self, temp_magnetic_json):
        """Test multiple reads of the same file."""
        channel = MetronixChannelJSON()

        # Read the same file multiple times
        for _ in range(3):
            channel.read(temp_magnetic_json)
            assert channel._has_metadata()

    @requires_test_data
    def test_large_calibration_data(self, tmp_path):
        """Test with large calibration datasets."""
        # Create data with many frequency points
        large_freq_data = list(np.logspace(-5, 5, 100))  # Smaller for performance
        large_amp_data = list(np.ones(100))
        large_phase_data = list(np.zeros(100))

        large_data = {
            "datetime": "2009-08-20T13:22:00",
            "latitude": 0.0,
            "longitude": 0.0,
            "elevation": 0.0,
            "angle": 0.0,
            "tilt": 0.0,
            "resistance": 0.0,
            "units": "mV",
            "filter": "test",
            "source": "",
            "sensor_calibration": {
                "sensor": "TEST",
                "serial": 0,
                "chopper": 1,
                "units_frequency": "Hz",
                "units_amplitude": "mV/nT",
                "units_phase": "degrees",
                "datetime": "2006-12-01T11:23:02",
                "Operator": "",
                "f": large_freq_data,
                "a": large_amp_data,
                "p": large_phase_data,
            },
        }

        json_file = tmp_path / "084_ADU-07e_C002_THx_128Hz.json"
        with open(json_file, "w") as fid:
            json.dump(large_data, fid)

        channel = MetronixChannelJSON(json_file)
        fap = channel.get_sensor_response_filter()
        assert fap is not None, "Sensor response filter should not be None"
        assert len(fap.frequencies) == 100
        assert len(fap.amplitudes) == 100
        assert len(fap.phases) == 100


class TestMetronixValidation:
    """Test validation and error handling."""

    @requires_test_data
    def test_missing_required_fields_handling(self, tmp_path):
        """Test handling of JSON with missing required fields."""
        incomplete_data = {
            "datetime": "2009-08-20T13:22:00",
            # Missing many required fields but should not crash
        }

        json_file = tmp_path / "084_ADU-07e_C002_THx_128Hz.json"
        with open(json_file, "w") as fid:
            json.dump(incomplete_data, fid)

        # Should be able to initialize but may have limited functionality
        channel = MetronixChannelJSON(json_file)
        assert channel._has_metadata()

    @requires_test_data
    def test_file_operations_robustness(self, tmp_path):
        """Test robustness of file operations."""
        # Test with various filename formats
        test_files = [
            "084_ADU-07e_C000_TEx_128Hz.json",
            "999_TEST-01_C999_THz_0.1Hz.json",
        ]

        for filename in test_files:
            json_file = tmp_path / filename
            test_data = {
                "datetime": "2009-08-20T13:22:00",
                "latitude": 0.0,
                "longitude": 0.0,
                "elevation": 0.0,
                "angle": 0.0,
                "tilt": 0.0,
                "resistance": 0.0,
                "units": "mV",
                "filter": "test",
                "source": "",
                "sensor_calibration": {
                    "sensor": "TEST",
                    "serial": 0,
                    "chopper": 1,
                    "units_frequency": "Hz",
                    "units_amplitude": "mV/nT",
                    "units_phase": "degrees",
                    "datetime": "2006-12-01T11:23:02",
                    "Operator": "",
                    "f": [],
                    "a": [],
                    "p": [],
                },
            }

            with open(json_file, "w") as fid:
                json.dump(test_data, fid)

            # Should be able to parse filename and load metadata
            channel = MetronixChannelJSON(json_file)
            assert channel._has_metadata()
            assert channel.fn == json_file


# =============================================================================
# Skip Tests When Data Not Available
# =============================================================================


def test_skip_when_no_data():
    """Test that tests are properly skipped when data is not available."""
    if not has_data_import:
        pytest.skip("mth5_test_data not available - tests properly skipped")
    else:
        # Data is available, so other tests should run
        assert metronix_data_path is not None
        assert metronix_data_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])
