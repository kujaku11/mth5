# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:30:08 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.timeseries.ts_helpers import (
    make_dt_coordinates,
    get_decimation_sample_rates,
)

# =============================================================================


class TestGetDecimationSampleRates(unittest.TestCase):
    def test_4096_to_1(self):
        self.assertListEqual(
            [512, 64, 8, 1], get_decimation_sample_rates(4096, 1, 8)
        )


class TestMakeDtCoordinates(unittest.TestCase):
    pass


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
