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
        parse_dates=["time"],
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
                "processing_run_01"
            )
        )
        self.decimation_level = self.fc_group.add_decimation_level("3")
        self.ds = read_fc_csv(csv_fn)
        self.decimation_level.from_xarray(self.ds)
        self.decimation_level.update_metadata()
        self.fc_group.update_metadata()

        self.expected_start = MTime(self.ds.time[0].values)
        self.expected_end = MTime(self.ds.time[-1].values)
        self.expected_window_step = 6144
        self.expected_sr_decimation_level = 0.015380859375
        self.expected_shape = (6, 64)
        self.expected_time = np.array(
            [
                "1980-01-01T00:00:00.000000000",
                "1980-01-01T01:42:24.000000000",
                "1980-01-01T03:24:48.000000000",
                "1980-01-01T05:07:12.000000000",
                "1980-01-01T06:49:36.000000000",
                "1980-01-01T08:32:00.000000000",
            ],
            dtype="datetime64[ns]",
        )

        self.expected_frequency = np.array(
            [
                0.0,
                0.00012207,
                0.00024414,
                0.00036621,
                0.00048828,
                0.00061035,
                0.00073242,
                0.00085449,
                0.00097656,
                0.00109863,
                0.0012207,
                0.00134277,
                0.00146484,
                0.00158691,
                0.00170898,
                0.00183105,
                0.00195312,
                0.0020752,
                0.00219727,
                0.00231934,
                0.00244141,
                0.00256348,
                0.00268555,
                0.00280762,
                0.00292969,
                0.00305176,
                0.00317383,
                0.0032959,
                0.00341797,
                0.00354004,
                0.00366211,
                0.00378418,
                0.00390625,
                0.00402832,
                0.00415039,
                0.00427246,
                0.00439453,
                0.0045166,
                0.00463867,
                0.00476074,
                0.00488281,
                0.00500488,
                0.00512695,
                0.00524902,
                0.00537109,
                0.00549316,
                0.00561523,
                0.0057373,
                0.00585938,
                0.00598145,
                0.00610352,
                0.00622559,
                0.00634766,
                0.00646973,
                0.0065918,
                0.00671387,
                0.00683594,
                0.00695801,
                0.00708008,
                0.00720215,
                0.00732422,
                0.00744629,
                0.00756836,
                0.00769043,
            ]
        )

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
            with self.subTest(f"{ch} sr_decimation_level"):
                self.assertEqual(
                    fc_ch.metadata.sample_rate_decimation_level,
                    self.expected_sr_decimation_level,
                )
            with self.subTest(f"{ch} shape"):
                self.assertTupleEqual(
                    fc_ch.hdf5_dataset.shape, self.expected_shape
                )

            with self.subTest(f"{ch} time"):
                self.assertTrue((fc_ch.time == self.expected_time).all())
            with self.subTest(f"{ch} frequency"):
                self.assertTrue(
                    np.isclose(fc_ch.frequency, self.expected_frequency).all()
                )

    def test_to_xarray(self):
        da = self.decimation_level.to_xarray()

        self.assertEqual(da, self.ds)

    def test_ch_to_xarray(self):
        fc_ch = self.decimation_level.get_channel("ex")
        ch_da = fc_ch.to_xarray()

        with self.subTest("time"):
            self.assertTrue((ch_da.time.values == self.expected_time).all())
        with self.subTest("frequency"):
            self.assertTrue(
                np.isclose(ch_da.frequency, self.expected_frequency).all()
            )
        with self.subTest("name"):
            self.assertTrue("ex", ch_da.name)

        with self.subTest("ex start"):
            self.assertEqual(
                ch_da.attrs["time_period.start"], self.expected_start
            )
        with self.subTest("ex end"):
            self.assertEqual(ch_da.attrs["time_period.end"], self.expected_end)
        with self.subTest("ex window_step"):
            self.assertEqual(
                ch_da.attrs["sample_rate_window_step"],
                self.expected_window_step,
            )
        with self.subTest("ex sr_decimation_level"):
            self.assertEqual(
                ch_da.attrs["sample_rate_decimation_level"],
                self.expected_sr_decimation_level,
            )
        with self.subTest("ex shape"):
            self.assertTupleEqual(ch_da.shape, self.expected_shape)

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.m.filename.unlink()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
