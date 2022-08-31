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

import numpy as np
from mth5.io.phoenix import read_phoenix, open_phoenix, PhoenixCollection

# =============================================================================


@unittest.skipIf(
    "peacock" in Path(__file__), "Only local files, cannot test in GitActions"
)
class TestReadPhoenixBinary(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.phx_obj = open_phoenix(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\Sample Data\10128_2021-04-27-025909\0\10128_60877DFD_0_00000001.bin"
        )

        self.data, self.footer = self.phx_obj.read_sequence()

        self.original = open_phoenix(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\Sample Data\10128_2021-04-27-025909\0\10128_60877DFD_0_00000001.bin"
        )
        self.original_data = self.original.read_frames()

    def test_readers_match(self):
        self.assertTrue(
            np.allclose(
                self.data[0 : self.original_data[0].size],
                self.original_data[0],
            )
        )
