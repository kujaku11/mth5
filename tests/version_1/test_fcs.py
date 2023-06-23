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

from mt_metadata.utils.mttime import MTime

# =============================================================================
fn_path = Path(__file__).parent
csv_fn = fn_path.joinpath("test1_dec_level_3.csv")


def read_fc_csv(csv_name):
    """
    read csv to xarray

    :param csv_name: DESCRIPTION
    :type csv_name: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    df = pd.read_csv(
        csv_name,
        index_col=[0, 1],
        parse_dates=[
            "time",
        ],
        skipinitialspace=True,
    )
    for col in df.columns:
        df[col] = np.complex128(df[col])

    return df.to_xarray()


class TestFCFromXarray(unittest.TestCase):
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
        self.decimation_level = self.fc_group.add_decimation_level("3")
        self.ds = read_fc_csv(csv_fn)
        self.expected_start = MTime(self.ds.time[0].values)
        self.expected_end = MTime(self.ds.time[-1].values)
        self.expected_window_step = 6144
        self.expected_sr_decimation_level = 0.015380859375
        self.decimation_level.from_xarray(self.ds)
        self.expected_shape = (64, 6)

    def test_channel_exists(self):
        self.assertListEqual(
            list(self.ds.data_vars.keys()),
            self.decimation_level.groups_list,
        )

    def test_channel_metadata(self):
        for ch in self.decimation_level.groups_list:
            fc_ch = self.decimation_level.get_channel(ch)
            with self.subTest(f"{ch} name"):
                self.assertEqual(fc_ch.metadata.component, ch)
            with self.subTest(f"{ch} start"):
                self.assertEqual(
                    fc_ch.metadata.time_period.start, self.expected_start
                )
            with self.subTest(f"{ch} end"):
                self.assertEqual(
                    fc_ch.metadata.time_period.end, self.expected_end
                )
            with self.subTest(f"{ch} window_step"):
                self.assertEqual(
                    fc_ch.metadata.sample_rate_window_step,
                    self.expected_window_step,
                )
            with self.subTest(f"{ch} window_step"):
                self.assertEqual(
                    fc_ch.metadata.sample_rate_decimation_level,
                    self.expected_sr_decimation_level,
                )
            with self.subTest(f"{ch} shape"):
                self.assertTupleEqual(
                    fc_ch.hdf5_dataset.shape, self.expected_shape
                )

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.m.filename.unlink()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
