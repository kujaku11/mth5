# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:04:17 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from collections import OrderedDict
from mth5.io.scripps.zenc import ZENC

# =============================================================================


class TestZENC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.channel_dict = OrderedDict(
            {
                "ex": {
                    "survey": "survey_01",
                    "station": "station_name",
                    "run": "run_name",
                    "channel": "channel_name",
                    "channel_number": 1,
                },
                "ey": {
                    "survey": "survey_01",
                    "station": "station_name",
                    "run": "run_name",
                    "channel": "channel_name",
                    "channel_number": 2,
                },
            }
        )

    def test_input_channel_map(self):
        z = ZENC(self.channel_dict)
        self.assertDictEqual(z.channel_map, self.channel_dict)

    def test_bad_input_channel_map_value(self):
        self.assertRaises(ValueError, ZENC, 8)

    def test_bad_input_channel_map_dict(self):
        self.assertRaises(KeyError, ZENC, {"ex": {"a": 8}})

    def test_input_channel_map_order(self):
        unsorted_channel_map = OrderedDict()
        unsorted_channel_map["ey"] = self.channel_dict["ey"]
        unsorted_channel_map["ex"] = self.channel_dict["ex"]
        z = ZENC(self.channel_dict)
        self.assertDictEqual(z.channel_map, self.channel_dict)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
