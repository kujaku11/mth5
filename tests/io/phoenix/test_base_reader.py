# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:35:56 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path
from collections import OrderedDict

import numpy as np
from mth5.io.phoenix.readers.base import TSReaderBase
from mth5.io.phoenix.readers.config import PhoenixConfig
from mth5.utils.helpers import get_compare_dict

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()),
    "Only local files, cannot test in GitActions",
)
class TestReadPhoenixContinuous(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = Path(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\Sample Data\10128_2021-04-27-032436\0\10128_608783F4_0_00000001.td_150"
        )
        self.phx_obj = TSReaderBase(self.fn)

        self.rxcal_fn = Path(__file__).parent.joinpath("example_rxcal.json")

        self.maxDiff = None

    def test_seq(self):
        self.assertEqual(self.phx_obj.seq, 1)

    def test_base_path(self):
        self.assertEqual(self.fn, self.phx_obj.base_path)

    def test_last_seq(self):
        self.assertEqual(self.phx_obj.last_seq, 2)

    def test_recording_id(self):
        self.assertEqual(self.phx_obj.recording_id, 1619493876)

    def test_channel_id(self):
        self.assertEqual(self.phx_obj.channel_id, 0)

    def test_recmeta_file_path(self):
        self.assertEqual(
            self.phx_obj.recmeta_file_path,
            Path(
                r"c:/Users/jpeacock/OneDrive - DOI/mt/phoenix_example_data/Sample Data/10128_2021-04-27-032436/recmeta.json"
            ),
        )

    def test_channel_map(self):
        self.assertDictEqual(
            self.phx_obj.channel_map,
            {0: "h2", 1: "e1", 2: "h1", 3: "h3", 4: "e2"},
        )

    def test_base_dir(self):
        self.assertEqual(self.phx_obj.base_dir, self.fn.parent)

    def test_file_name(self):
        self.assertEqual(self.phx_obj.file_name, self.fn.name)

    def test_file_extension(self):
        self.assertEqual(self.phx_obj.file_extension, self.fn.suffix)

    def test_instrument_id(self):
        self.assertEqual(self.phx_obj.instrument_id, "10128")

    def test_file_size(self):
        self.assertEqual(self.phx_obj.file_size, 215528)

    def test_max_samples(self):
        self.assertEqual(self.phx_obj.max_samples, 53850)

    def test_sequence_list(self):
        self.assertListEqual(
            self.phx_obj.sequence_list,
            [
                Path(
                    "c:/Users/jpeacock/OneDrive - DOI/mt/phoenix_example_data/Sample Data/10128_2021-04-27-032436/0/10128_608783F4_0_00000001.td_150"
                ),
                Path(
                    "c:/Users/jpeacock/OneDrive - DOI/mt/phoenix_example_data/Sample Data/10128_2021-04-27-032436/0/10128_608783F4_0_00000002.td_150"
                ),
            ],
        )

    def test_config_file_path(self):
        self.assertEqual(
            self.phx_obj.config_file_path,
            Path(
                "c:/Users/jpeacock/OneDrive - DOI/mt/phoenix_example_data/Sample Data/10128_2021-04-27-032436/config.json"
            ),
        )

    def test_get_config_object(self):
        self.assertIsInstance(self.phx_obj.get_config_object(), PhoenixConfig)

    def test_get_lowpass_filter_name(self):
        self.assertEqual(self.phx_obj.get_lowpass_filter_name(), 10000)

    def test_has_header(self):
        self.assertEqual(self.phx_obj._has_header(), True)

    def test_channel_metadata(self):
        ch = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "h2"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [False, False]),
                (
                    "filter.name",
                    ["mtu-5c_rmt03_10128_10000hz_low_pass", "coil_0_response"],
                ),
                ("location.elevation", 140.10263061523438),
                ("location.latitude", 43.69625473022461),
                ("location.longitude", -79.39364624023438),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 150.0),
                ("sensor.id", "0"),
                ("sensor.manufacturer", "Phoenix Geophysics"),
                ("sensor.model", "MTC-150"),
                ("sensor.type", "4"),
                ("time_period.end", "2021-04-27T03:30:43+00:00"),
                ("time_period.start", "2021-04-26T20:24:18+00:00"),
                ("type", "magnetic"),
                ("units", "volts"),
            ]
        )

        self.assertDictEqual(
            self.phx_obj.channel_metadata.to_dict(single=True), ch
        )

    def test_run_metadata(self):
        rn = OrderedDict(
            [
                ("channels_recorded_auxiliary", []),
                ("channels_recorded_electric", []),
                ("channels_recorded_magnetic", ["h2"]),
                ("data_logger.firmware.author", None),
                ("data_logger.firmware.name", None),
                ("data_logger.firmware.version", "00010036X"),
                ("data_logger.id", "10128"),
                ("data_logger.manufacturer", "Phoenix Geophysics"),
                ("data_logger.model", "MTU-5C"),
                ("data_logger.power_source.voltage.start", 12.475),
                ("data_logger.timing_system.drift", -2.0),
                ("data_logger.timing_system.type", "GPS"),
                ("data_logger.timing_system.uncertainty", 201.0),
                ("data_logger.type", "RMT03"),
                ("data_type", "BBMT"),
                ("id", "sr150_0001"),
                ("sample_rate", 150.0),
                ("time_period.end", "2021-04-27T03:30:43+00:00"),
                ("time_period.start", "2021-04-26T20:24:18+00:00"),
            ]
        )

        self.assertDictEqual(self.phx_obj.run_metadata.to_dict(single=True), rn)

    def test_station_metadata(self):
        st = OrderedDict(
            [
                ("acquired_by.name", "J"),
                ("acquired_by.organization", "Phoenix Geophysics"),
                ("channels_recorded", ["h2"]),
                ("data_type", "BBMT"),
                ("geographic_name", None),
                ("id", "Masked Cordinates"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 0.0),
                ("location.elevation", 181.129387),
                ("location.latitude", 43.69602),
                ("location.longitude", -79.393771),
                ("orientation.method", None),
                ("orientation.reference_frame", "geographic"),
                ("provenance.archive.name", None),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.creator.name", None),
                ("provenance.software.author", None),
                ("provenance.software.name", None),
                ("provenance.software.version", None),
                ("provenance.submitter.email", None),
                ("provenance.submitter.name", None),
                ("provenance.submitter.organization", None),
                ("release_license", "CC0-1.0"),
                ("run_list", ["sr150_0001"]),
                ("time_period.end", "2021-04-27T03:30:43+00:00"),
                ("time_period.start", "2021-04-26T20:24:18+00:00"),
            ]
        )

        self.assertDictEqual(
            self.phx_obj.station_metadata.to_dict(single=True), st
        )

    def test_get_dipole_filter(self):
        self.assertEqual(self.phx_obj.get_dipole_filter(), None)

    def test_get_v_to_mv_filter(self):
        f = OrderedDict(
            [
                ("calibration_date", "1980-01-01"),
                ("gain", 1000.0),
                ("name", "v_to_mv"),
                ("type", "coefficient"),
                ("units_in", "V"),
                ("units_out", "mV"),
            ]
        )

        self.assertDictEqual(
            self.phx_obj.get_v_to_mv_filter().to_dict(single=True), f
        )

    def test_get_channel_response_filter(self):
        cr = self.phx_obj.get_channel_response_filter()

        with self.subTest("length"):
            self.assertEqual(len(cr.filters_list), 1)
        with self.subTest("names"):
            self.assertEqual(cr.filters_list[0].name, "v_to_mv")


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
