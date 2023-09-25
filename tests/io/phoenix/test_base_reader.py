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
        self.assertEqual(self.phx_obj._has_header(), False)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
