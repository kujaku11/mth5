# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:39:30 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path
from collections import OrderedDict

from mth5.io.phoenix import open_phoenix
from mth5.utils.helpers import get_compare_dict

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()),
    "Only local files, cannot test in GitActions",
)
class TestReadPhoenixSegmented(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.phx_obj = open_phoenix(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\Sample Data\10128_2021-04-27-032436\0\10128_608783F4_0_00000001.td_24k"
        )

        self.segment = self.phx_obj.read_segment()

        self.rxcal_fn = Path(__file__).parent.joinpath("example_rxcal.json")
        self.maxDiff = None

    def test_subheader(self):
        true_dict = {
            "channel_id": 0,
            "channel_type": "H",
            "elevation": 140.10263061523438,
            "gps_time_stamp": "2021-04-27T03:24:42+00:00",
            "header_length": 32,
            "instrument_serial_number": "10128",
            "instrument_type": "MTU-5C",
            "latitude": 43.69625473022461,
            "longitude": -79.39364624023438,
            "missing_count": 0,
            "n_samples": 48000,
            "sample_rate": 24000.0,
            "saturation_count": 0,
            "segment": 0,
            "segment_end_time": "2021-04-27T03:24:44+00:00",
            "segment_start_time": "2021-04-27T03:24:42+00:00",
            "value_max": 0.24964138865470886,
            "value_mean": -1.3566585039370693e-05,
            "value_min": 3.4028234663852886e38,
        }

        for key, original_value in true_dict.items():
            new_value = getattr(self.segment, key)
            with self.subTest(f"{key}"):
                if isinstance(original_value, (list)):
                    self.assertListEqual(original_value, new_value)
                elif isinstance(original_value, float):
                    self.assertAlmostEqual(original_value, new_value)
                else:
                    self.assertEqual(original_value, new_value)

    def test_attributes(self):
        true_dict = {
            "ad_plus_minus_range": 5.0,
            "attenuator_gain": 1.0,
            "base_dir": Path(
                "c:/Users/jpeacock/OneDrive - DOI/mt/phoenix_example_data/Sample Data/10128_2021-04-27-032436/0"
            ),
            "base_path": Path(
                "c:/Users/jpeacock/OneDrive - DOI/mt/phoenix_example_data/Sample Data/10128_2021-04-27-032436/0/10128_608783F4_0_00000001.td_24k"
            ),
            "battery_voltage_v": 12.475,
            "board_model_main": "BCM01",
            "board_model_revision": "",
            "bytes_per_sample": 4,
            "ch_board_model": "BCM01-I",
            "ch_board_serial": 200803,
            "ch_firmware": 65567,
            "channel_id": 0,
            "channel_main_gain": 4.0,
            "channel_map": {0: "h2", 1: "e1", 2: "h1", 3: "h3", 4: "e2"},
            "channel_type": "H",
            "data_footer": 0,
            "decimation_node_id": 0,
            "detected_channel_type": "H",
            "file_extension": ".td_24k",
            "file_name": "10128_608783F4_0_00000001.td_24k",
            "file_sequence": 1,
            "file_size": 2304512,
            "file_type": 2,
            "file_version": 2,
            "frag_period": 360,
            "frame_rollover_count": 0,
            "frame_size": 64,
            "frame_size_bytes": 64,
            "future1": 32,
            "future2": 0,
            "gps_elevation": 140.10263061523438,
            "gps_horizontal_accuracy": 17.512,
            "gps_lat": 43.69625473022461,
            "gps_long": -79.39364624023438,
            "gps_vertical_accuracy": 22.404,
            "hardware_configuration": (4, 3, 0, 0, 0, 9, 128, 0),
            "header_length": 128,
            "instrument_id": "10128",
            "instrument_serial_number": "10128",
            "instrument_type": "MTU-5C",
            "intrinsic_circuitry_gain": 1.0,
            "last_seq": 2,
            "lp_frequency": 10000,
            "max_samples": 576096,
            "max_signal": 2.0711588859558105,
            "min_signal": -2.0549893379211426,
            "missing_frames": 0,
            "preamp_gain": 1.0,
            "recording_id": 1619493876,
            "recording_start_time": "2021-04-26T20:24:18+00:00",
            "report_hw_sat": False,
            "sample_rate": 24000,
            "sample_rate_base": 24000,
            "sample_rate_exp": 0,
            "saturated_frames": 0,
            "seq": 1,
            "sequence_list": [
                Path(
                    "c:/Users/jpeacock/OneDrive - DOI/mt/phoenix_example_data/Sample Data/10128_2021-04-27-032436/0/10128_608783F4_0_00000001.td_24k"
                ),
                Path(
                    "c:/Users/jpeacock/OneDrive - DOI/mt/phoenix_example_data/Sample Data/10128_2021-04-27-032436/0/10128_608783F4_0_00000002.td_24k"
                ),
            ],
            "timing_flags": 55,
            "timing_sat_count": 7,
            "timing_stability": 201,
            "timing_status": (55, 7, 201),
            "total_circuitry_gain": 4.0,
            "total_selectable_gain": 4.0,
        }

        for key, original_value in true_dict.items():
            new_value = getattr(self.phx_obj, key)
            with self.subTest(f"{key}"):
                if isinstance(original_value, (list)):
                    self.assertListEqual(original_value, new_value)
                elif isinstance(original_value, float):
                    self.assertAlmostEqual(original_value, new_value)
                else:
                    self.assertEqual(original_value, new_value)

    def test_to_channel_ts(self):
        ch_ts = self.phx_obj.to_channel_ts(rxcal_fn=self.rxcal_fn)

        ch_metadata = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "h2"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [False, False]),
                (
                    "filter.name",
                    ["mtu-5c_rmt03-j_666_h2_10000hz_lowpass", "v_to_mv"],
                ),
                ("location.elevation", 140.10263061523438),
                ("location.latitude", 43.69625473022461),
                ("location.longitude", -79.39364624023438),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 24000.0),
                ("sensor.id", "0"),
                ("sensor.manufacturer", "Phoenix Geophysics"),
                ("sensor.model", "MTC-150"),
                ("sensor.type", "4"),
                ("time_period.end", "2021-04-27T03:25:13.999958333+00:00"),
                ("time_period.start", "2021-04-27T03:25:12+00:00"),
                ("type", "magnetic"),
                ("units", "volts"),
            ]
        )

        for key, value in ch_metadata.items():
            with self.subTest(key):
                if isinstance(value, float):
                    self.assertAlmostEqual(
                        value,
                        ch_ts.channel_metadata.get_attr_from_name(key),
                        5,
                    )

                else:
                    self.assertEqual(
                        value, ch_ts.channel_metadata.get_attr_from_name(key)
                    )

        with self.subTest("channel_response_filter_length"):
            self.assertEqual(2, len(ch_ts.channel_response_filter.filters_list))

        with self.subTest("channel_response_filter_frequency_shape"):
            self.assertEqual(
                (69,),
                ch_ts.channel_response_filter.filters_list[0].frequencies.shape,
            )

        with self.subTest("Channel Size"):
            self.assertEqual(48000, ch_ts.ts.size)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
