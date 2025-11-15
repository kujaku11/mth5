# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:39:30 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from mth5.io.phoenix import open_phoenix


try:
    import mth5_test_data

    phx_data_path = mth5_test_data.get_test_data_path("phoenix") / "sample_data"
    has_test_data = True
except ImportError:
    has_test_data = False

# =============================================================================


class TestReadPhoenixNative(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.phx_obj = open_phoenix(
            phx_data_path
            / "10128_2021-04-27-025909"
            / "0"
            / "10128_60877DFD_0_00000001.bin"
        )

        self.data, self.footer = self.phx_obj.read_sequence()

        self.original = open_phoenix(
            phx_data_path
            / "10128_2021-04-27-025909"
            / "0"
            / "10128_60877DFD_0_00000001.bin"
        )
        self.original_data = self.original.read_frames(10)

        self.rxcal_fn = phx_data_path / "example_rxcal.json"
        self.maxDiff = None

    def test_readers_match(self):
        self.assertTrue(
            np.allclose(
                self.data[0 : self.original_data[0].size],
                self.original_data[0],
            )
        )

    def test_attributes(self):
        true_dict = {
            "ad_plus_minus_range": 5.0,
            "attenuator_gain": 1.0,
            "base_dir": phx_data_path / "10128_2021-04-27-025909" / "0",
            "base_path": phx_data_path
            / "10128_2021-04-27-025909"
            / "0"
            / "10128_60877DFD_0_00000001.bin",
            "battery_voltage_v": 12.446,
            "board_model_main": "BCM01",
            "board_model_revision": "I",
            "bytes_per_sample": 3,
            "ch_board_model": "BCM01-I",
            "ch_board_serial": 200803,
            "ch_firmware": 65567,
            "channel_id": 0,
            "channel_main_gain": 4.0,
            "channel_map": {0: "h2", 1: "e1", 2: "h1", 3: "h3", 4: "e2"},
            "channel_type": "H",
            "data_footer": 0,
            "data_scaling": 1,
            "decimation_node_id": 0,
            "detected_channel_type": "H",
            "file_extension": ".bin",
            "file_name": "10128_60877DFD_0_00000001.bin",
            "file_sequence": 2,
            "file_size": 4608128,
            "file_type": 1,
            "file_version": 3,
            "footer_idx_samp_mask": 268435455,
            "footer_sat_mask": 1879048192,
            "frag_period": 60,
            "frame_rollover_count": 0,
            "frame_size": 64,
            "frame_size_bytes": 64,
            "future1": 28,
            "future2": 0,
            "gps_elevation": 70.11294555664062,
            "gps_horizontal_accuracy": 11.969,
            "gps_lat": 43.69640350341797,
            "gps_long": -79.3936996459961,
            "gps_vertical_accuracy": 38.042,
            "hardware_configuration": (4, 3, 0, 0, 0, 10, 128, 0),
            "header_length": 128,
            "input_plusminus_range": 1.25,
            "instrument_id": "10128",
            "instrument_serial_number": "10128",
            "instrument_type": "MTU-5C",
            "intrinsic_circuitry_gain": 1.0,
            "last_frame": 0,
            "last_seq": 2,
            "lp_frequency": 10000,
            "max_samples": 1152000,
            "max_signal": 2.0269203186035156,
            "min_signal": -2.0260202884674072,
            "missing_frames": 0,
            "npts_per_frame": 20,
            "preamp_gain": 1.0,
            "recording_id": 1619492349,
            "recording_start_time": "2021-04-27T02:58:51+00:00",
            "report_hw_sat": False,
            "sample_rate": 24000,
            "sample_rate_base": 24000,
            "sample_rate_exp": 0,
            "saturated_frames": 0,
            "scale_factor": 2.3283064365386963e-09,
            "seq": 1,
            "sequence_list": [
                phx_data_path
                / "10128_2021-04-27-025909"
                / "0"
                / "10128_60877DFD_0_00000001.bin",
                phx_data_path
                / "10128_2021-04-27-025909"
                / "0"
                / "10128_60877DFD_0_00000002.bin",
                phx_data_path
                / "10128_2021-04-27-025909"
                / "0"
                / "10128_60877DFD_0_00000003.bin",
            ],
            "timing_flags": 55,
            "timing_sat_count": 6,
            "timing_stability": 145,
            "timing_status": (55, 6, 145),
            "total_circuitry_gain": 4.0,
            "total_selectable_gain": 4.0,
        }

        for key, original_value in true_dict.items():
            new_value = getattr(self.phx_obj, key)
            with self.subTest(f"test {key}"):
                if isinstance(original_value, (list)):
                    self.assertListEqual(original_value, new_value)
                elif isinstance(original_value, float):
                    self.assertAlmostEqual(original_value, new_value)
                else:
                    self.assertEqual(original_value, new_value)


class TestReadPhoenixNativeToChannelTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.phx_obj = open_phoenix(
            phx_data_path
            / "10128_2021-04-27-025909"
            / "0"
            / "10128_60877DFD_0_00000001.bin"
        )

        self.rxcal_fn = Path(__file__).parent.joinpath("example_rxcal.json")
        self.maxDiff = None

        self.ch_ts = self.phx_obj.to_channel_ts(rxcal_fn=self.rxcal_fn)

    def test_metadata(self):
        ch_metadata = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "h2"),
                ("data_quality.rating.value", None),
                # ("filter.applied", [True, True]),
                # (
                #     "filter.name",
                #     ["mtu-5c_rmt03_10128_h2_10000hz_lowpass", "v_to_mv"],
                # ),
                ("location.elevation", 70.11294555664062),
                ("location.latitude", 43.69640350341797),
                ("location.longitude", -79.3936996459961),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 24000.0),
                ("sensor.id", "0"),
                ("sensor.manufacturer", "Phoenix Geophysics"),
                ("sensor.model", "MTC-150"),
                ("sensor.type", "4"),
                ("time_period.end", "2021-04-27T03:01:50.999958333+00:00"),
                ("time_period.start", "2021-04-27T02:58:51+00:00"),
                ("type", "magnetic"),
                ("units", "Volt"),
            ]
        )

        for key, value in ch_metadata.items():
            with self.subTest(key):
                if isinstance(value, float):
                    self.assertAlmostEqual(
                        value,
                        self.ch_ts.channel_metadata.get_attr_from_name(key),
                        5,
                    )

                else:
                    self.assertEqual(
                        value,
                        self.ch_ts.channel_metadata.get_attr_from_name(key),
                    )

        with self.subTest("filters_exist"):
            filters = self.ch_ts.channel_metadata.get_attr_from_name("filters")
            self.assertIsInstance(filters, list)
            self.assertEqual(len(filters), 3)
            # Check that each filter is an AppliedFilter object with expected names
            expected_names = [
                "mtu-5c_rmt03_10128_h2_10000hz_lowpass",
                "v_to_mv",
                "coil_0_response",
            ]
            for filt, expected_name in zip(filters, expected_names):
                self.assertEqual(filt.name, expected_name)

    def test_channel_response_length(self):
        self.assertEqual(2, len(self.ch_ts.channel_response.filters_list))

    def test_channel_response_frequency_shape(self):
        self.assertEqual(
            (69,),
            self.ch_ts.channel_response.filters_list[0].frequencies.shape,
        )

    def test_channel_size(self):
        self.assertEqual(4320000, self.ch_ts.ts.size)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
