# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:59:26 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
from mth5.mth5 import MTH5

# =============================================================================
fn_path = Path(__file__).parent


def create_nd_array(ch, n_samples):
    nd_array = np.zeros(
        n_samples,
        dtype=[
            ("time", "S32"),
            ("frequency", float),
            (ch, complex),
        ],
    )
    nd_array["time"] = pd.date_range(
        "2020-01-01T00:00:00", periods=n_samples, freq="s"
    )
    nd_array["frequency"] = np.logspace(-5, 5, 50)
    nd_array[ch] = np.arange(n_samples) + 1j * np.arange(n_samples)
    return nd_array


class TestFC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = MTH5()
        self.m.file_version = "0.1.0"
        self.m.open_mth5(fn_path.joinpath("fc_test.h5"))
        self.station_group = self.m.add_station("mt01")
        self.fc_group = (
            self.station_group.fourier_coefficients_group.add_fc_group(
                "default"
            )
        )
        self.decimation_level = self.fc_group.add_decimation_level("1")
        self.n_samples = 50

    def test_np_structured_array_input(self):
        name = "nd_array"
        a = create_nd_array(name, self.n_samples)
        ch = self.decimation_level.add_channel(name, fc_data=a)

        with self.subTest("channel_exists"):
            self.assertIn(name, self.decimation_level.groups_list)

        with self.subTest("channel_name"):
            self.assertEqual(ch.metadata.name, name)

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.m.filename.unlink()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
