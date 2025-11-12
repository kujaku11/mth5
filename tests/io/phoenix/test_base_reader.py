# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:35:56 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from collections import OrderedDict
from types import SimpleNamespace

from mth5.io.phoenix.readers.base import TSReaderBase
from mth5.io.phoenix.readers.config import PhoenixConfig


try:
    import mth5_test_data

    phx_data_path = mth5_test_data.get_test_data_path("phoenix") / "sample_data"
    has_test_data = True
except ImportError:
    has_test_data = False


# =============================================================================


@unittest.skipIf(
    has_test_data is False,
    "Only local files, cannot test in GitActions",
)
class TestReadPhoenixContinuous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fn = (
            phx_data_path
            / "10128_2021-04-27-032436"
            / "0"
            / "10128_608783F4_0_00000001.td_150"
        )
        cls.phx_obj = TSReaderBase(cls.fn)

        cls.rxcal_fn = phx_data_path / "example_rxcal.json"

        cls.maxDiff = None

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
            phx_data_path / "10128_2021-04-27-032436" / "recmeta.json",
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
                phx_data_path
                / "10128_2021-04-27-032436"
                / "0"
                / "10128_608783F4_0_00000001.td_150",
                phx_data_path
                / "10128_2021-04-27-032436"
                / "0"
                / "10128_608783F4_0_00000002.td_150",
            ],
        )

    def test_config_file_path(self):
        # Test that config file path exists and has correct name
        self.assertTrue(str(self.phx_obj.config_file_path).endswith("config.json"))

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
                ("component", "hx"),
                ("data_quality.rating.value", None),
                (
                    "filters",
                    [
                        {
                            "applied_filter": OrderedDict(
                                [
                                    ("applied", True),
                                    ("name", "mtu-5c_rmt03_10128_10000hz_low_pass"),
                                    ("stage", 1),
                                ]
                            )
                        },
                        {
                            "applied_filter": OrderedDict(
                                [
                                    ("applied", True),
                                    ("name", "coil_0_response"),
                                    ("stage", 2),
                                ]
                            )
                        },
                    ],
                ),
                ("h_field_max.end", 0.0),
                ("h_field_max.start", 0.0),
                ("h_field_min.end", 0.0),
                ("h_field_min.start", 0.0),
                ("location.datum", "WGS 84"),
                ("location.elevation", 140.10263061523438),
                ("location.latitude", 43.69625473022461),
                ("location.longitude", -79.39364624023438),
                ("location.x", 0.0),
                ("location.y", 0.0),
                ("location.z", 0.0),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 150.0),
                ("sensor.manufacturer", "Phoenix Geophysics"),
                ("time_period.end", "2021-04-27T03:30:43+00:00"),
                ("time_period.start", "2021-04-27T03:24:18+00:00"),
                ("type", "magnetic"),
                ("units", "Volt"),
            ]
        )

        self.assertDictEqual(self.phx_obj.channel_metadata.to_dict(single=True), ch)

    def test_run_metadata(self):
        rn = OrderedDict(
            [
                ("acquired_by.author", ""),
                ("channels_recorded_auxiliary", []),
                ("channels_recorded_electric", []),
                ("channels_recorded_magnetic", ["h2", "hx"]),
                ("data_logger.firmware.author", ""),
                ("data_logger.firmware.name", ""),
                ("data_logger.firmware.version", "00010036X"),
                ("data_logger.id", "10128"),
                ("data_logger.manufacturer", "Phoenix Geophysics"),
                ("data_logger.model", "MTU-5C"),
                ("data_logger.power_source.voltage.end", 0.0),
                ("data_logger.power_source.voltage.start", 12.475),
                ("data_logger.timing_system.drift", -2.0),
                ("data_logger.timing_system.type", "GPS"),
                ("data_logger.timing_system.uncertainty", 201.0),
                ("data_logger.type", "RMT03"),
                ("data_type", "BBMT"),
                ("id", "sr150_0001"),
                ("metadata_by.author", ""),
                ("provenance.archive.name", ""),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.creator.author", ""),
                ("provenance.software.author", ""),
                ("provenance.software.name", ""),
                ("provenance.software.version", ""),
                ("provenance.submitter.author", ""),
                ("sample_rate", 150.0),
                ("time_period.end", "2021-04-27T03:30:43+00:00"),
                ("time_period.start", "2021-04-27T03:24:18+00:00"),
            ]
        )

        self.assertDictEqual(self.phx_obj.run_metadata.to_dict(single=True), rn)

    def test_station_metadata(self):
        st = OrderedDict(
            [
                ("acquired_by.author", ""),
                ("acquired_by.organization", "Phoenix Geophysics"),
                ("channel_layout", "X"),
                ("channels_recorded", ["h2", "hx"]),
                ("data_type", "BBMT"),
                ("geographic_name", ""),
                ("id", "Masked_Cordinates"),
                ("location.datum", "WGS 84"),
                ("location.declination.model", "IGRF"),
                ("location.declination.value", 0.0),
                ("location.elevation", 181.129387),
                ("location.latitude", 43.69602),
                ("location.longitude", -79.393771),
                ("location.x", 0.0),
                ("location.y", 0.0),
                ("location.z", 0.0),
                ("orientation.method", "compass"),
                ("orientation.reference_frame", "geographic"),
                ("orientation.value", "orthogonal"),
                ("provenance.archive.name", ""),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.creator.author", ""),
                ("provenance.software.author", ""),
                ("provenance.software.name", ""),
                ("provenance.software.version", ""),
                ("provenance.submitter.author", ""),
                ("run_list", ["sr150_0001"]),
                ("time_period.end", "2021-04-27T03:30:43+00:00"),
                ("time_period.start", "2021-04-27T03:24:18+00:00"),
            ]
        )

        self.assertDictEqual(self.phx_obj.station_metadata.to_dict(single=True), st)

    def test_get_dipole_filter(self):
        self.assertEqual(self.phx_obj.get_dipole_filter(), None)

    def test_get_v_to_mv_filter(self):
        f = OrderedDict(
            [
                ("gain", 1000.0),
                ("name", "v_to_mv"),
                ("sequence_number", 0),
                ("type", "coefficient"),
                ("units_in", "Volt"),
                ("units_out", "milliVolt"),
            ]
        )

        self.assertDictEqual(self.phx_obj.get_v_to_mv_filter().to_dict(single=True), f)

    def test_get_channel_response(self):
        cr = self.phx_obj.get_channel_response()

        with self.subTest("length"):
            self.assertEqual(len(cr.filters_list), 1)
        with self.subTest("names"):
            self.assertEqual(cr.filters_list[0].name, "v_to_mv")

    def test_rx_metadata_obj(self):
        self.assertIsInstance(self.phx_obj.rx_metadata.obj, SimpleNamespace)

    def test_rx_metadata_emap(self):
        self.assertDictEqual(
            self.phx_obj.rx_metadata._e_map,
            {
                "tag": "component",
                "ty": "type",
                "ga": "gain",
                "sampleRate": "sample_rate",
                "pot_p": "contact_resistance.start",
                "pot_n": "contact_resistance.end",
            },
        )

    def test_rx_metadata_hmap(self):
        self.assertDictEqual(
            self.phx_obj.rx_metadata._h_map,
            {
                "tag": "component",
                "ty": "type",
                "ga": "gain",
                "sampleRate": "sample_rate",
                "type_name": "sensor.model",
                "type": "sensor.type",
                "serial": "sensor.id",
            },
        )

    def test_rx_metadata_fn(self):
        # Test that recmeta file path exists and has correct name
        self.assertTrue(str(self.phx_obj.rx_metadata.fn).endswith("recmeta.json"))

    def test_rx_metadata_e_metadata(self):
        for ch in ["e1", "e2"]:
            with self.subTest(ch):
                self.assertEqual(
                    self.phx_obj.rx_metadata._to_electric_metadata(ch),
                    getattr(self.phx_obj.rx_metadata, f"{ch}_metadata"),
                )

    def test_rx_metadata_h_metadata(self):
        for ch in ["h1", "h2", "h3"]:
            with self.subTest(ch):
                self.assertEqual(
                    self.phx_obj.rx_metadata._to_magnetic_metadata(ch),
                    getattr(self.phx_obj.rx_metadata, f"{ch}_metadata"),
                )

    def test_rx_metadata_run_metadata(self):
        rn = OrderedDict(
            [
                ("acquired_by.author", ""),
                ("channels_recorded_auxiliary", []),
                ("channels_recorded_electric", []),
                ("channels_recorded_magnetic", []),
                ("data_logger.firmware.author", ""),
                ("data_logger.firmware.name", ""),
                ("data_logger.firmware.version", "00010036X"),
                ("data_logger.model", "MTU-5C"),
                ("data_logger.power_source.voltage.end", 0.0),
                ("data_logger.power_source.voltage.start", 0.0),
                ("data_logger.timing_system.drift", -2.0),
                ("data_logger.timing_system.type", "GPS"),
                ("data_logger.timing_system.uncertainty", 0.0),
                ("data_logger.type", "RMT03"),
                ("data_type", "BBMT"),
                ("id", ""),
                ("metadata_by.author", ""),
                ("provenance.archive.name", ""),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.creator.author", ""),
                ("provenance.software.author", ""),
                ("provenance.software.name", ""),
                ("provenance.software.version", ""),
                ("provenance.submitter.author", ""),
                ("sample_rate", 0.0),
                ("time_period.end", "1980-01-01T00:00:00+00:00"),
                ("time_period.start", "1980-01-01T00:00:00+00:00"),
            ]
        )

        self.assertDictEqual(
            self.phx_obj.rx_metadata.run_metadata.to_dict(single=True), rn
        )

    def test_rx_metadata_station_metadata(self):
        st = OrderedDict(
            [
                ("acquired_by.author", ""),
                ("acquired_by.organization", "Phoenix Geophysics"),
                ("channel_layout", "X"),
                ("channels_recorded", []),
                ("data_type", "BBMT"),
                ("geographic_name", ""),
                ("id", "Masked_Cordinates"),
                ("location.datum", "WGS 84"),
                ("location.declination.model", "IGRF"),
                ("location.declination.value", 0.0),
                ("location.elevation", 181.129387),
                ("location.latitude", 43.69602),
                ("location.longitude", -79.393771),
                ("location.x", 0.0),
                ("location.y", 0.0),
                ("location.z", 0.0),
                ("orientation.method", "compass"),
                ("orientation.reference_frame", "geographic"),
                ("orientation.value", "orthogonal"),
                ("provenance.archive.name", ""),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.creator.author", ""),
                ("provenance.software.author", ""),
                ("provenance.software.name", ""),
                ("provenance.software.version", ""),
                ("provenance.submitter.author", ""),
                ("run_list", []),
                ("time_period.end", "1980-01-01T00:00:00+00:00"),
                ("time_period.start", "1980-01-01T00:00:00+00:00"),
            ]
        )

        self.assertDictEqual(
            self.phx_obj.rx_metadata.station_metadata.to_dict(single=True), st
        )


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
