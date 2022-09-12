# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 21:58:30 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.utils import fdsn_tools
from mt_metadata.timeseries import Electric, Magnetic, Auxiliary

# =============================================================================


class TestFDSNTools(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.ex = Electric(
            component="ex",
            channel_number=1,
            sample_rate=50,
            measurement_azimuth=5,
        )
        self.hx = Magnetic(
            component="hx",
            channel_number=1,
            sample_rate=50,
            measurement_azimuth=95,
        )
        self.tx = Auxiliary(
            component="temp",
            channel_number=1,
            sample_rate=50,
            measurement_azimuth=105,
        )

    def test_location_code(self):
        pass


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
