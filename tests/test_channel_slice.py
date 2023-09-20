# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:33:28 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

import numpy as np
from mth5.mth5 import MTH5
from mt_metadata.timeseries import Electric

# =============================================================================


class TestMTH5ChannelSlice(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = MTH5(file_version="0.1.0")
        self.m.open_mth5("test.h5", "w")
        self.station_group = self.m.add_station("mt01")
        self.run_group = self.station_group.add_run("a")
        channel_metadata = Electric(component="ex", sample_rate=1)
        self.n_samples = 4096
        self.ch_ds = self.run_group.add_channel(
            "ex",
            "electric",
            np.arange(self.n_samples),
            channel_metadata=channel_metadata,
        )

    def test_get_index_from_time_start(self):
        self.assertEqual(0, self.ch_ds.get_index_from_time(self.ch_ds.start))

    def test_get_index_from_time_start_too_early(self):
        self.assertEqual(
            -5, self.ch_ds.get_index_from_time(self.ch_ds.start - 5)
        )

    def test_get_index_from_time_end(self):
        self.assertEqual(
            self.n_samples, self.ch_ds.get_index_from_end_time(self.ch_ds.end)
        )

    def test_get_slice_index_values_times(self):
        start_index, end_index, npts = self.ch_ds._get_slice_index_values(
            self.ch_ds.start, end=self.ch_ds.end
        )
        with self.subTest("npts"):
            self.assertEqual(npts, self.n_samples)
        with self.subTest("start"):
            self.assertEqual(start_index, 0)
        with self.subTest("end"):
            self.assertEqual(end_index, self.n_samples)

    def test_get_slice_index_values_n_samples(self):
        start_index, end_index, npts = self.ch_ds._get_slice_index_values(
            self.ch_ds.start, n_samples=self.n_samples
        )
        with self.subTest("npts"):
            self.assertEqual(npts, self.n_samples)
        with self.subTest("start"):
            self.assertEqual(start_index, 0)
        with self.subTest("end"):
            self.assertEqual(end_index, self.n_samples)

    def test_full_slice_from_time(self):
        data = self.ch_ds.time_slice(self.ch_ds.start, end=self.ch_ds.end)
        with self.subTest("size"):
            self.assertEqual(data.data_array.size, self.n_samples)
        with self.subTest("start"):
            self.assertEqual(data.start, self.ch_ds.start)
        with self.subTest("end"):
            self.assertEqual(data.end, self.ch_ds.end)

    def test_full_slice_overtime(self):
        data = self.ch_ds.time_slice(self.ch_ds.start, end=self.ch_ds.end + 5)
        with self.subTest("size"):
            self.assertEqual(data.data_array.size, self.n_samples)
        with self.subTest("start"):
            self.assertEqual(data.start, self.ch_ds.start)
        with self.subTest("end"):
            self.assertEqual(data.end, self.ch_ds.end)

    def test_full_slice_from_points(self):
        data = self.ch_ds.time_slice(
            self.ch_ds.start, n_samples=self.n_samples
        )
        with self.subTest("size"):
            self.assertEqual(data.data_array.size, self.n_samples)
        with self.subTest("start"):
            self.assertEqual(data.start, self.ch_ds.start)
        with self.subTest("end"):
            self.assertEqual(data.end, self.ch_ds.end)

    def test_full_slice_from_too_many_points(self):
        data = self.ch_ds.time_slice(self.ch_ds.start, n_samples=5096)
        with self.subTest("size"):
            self.assertEqual(data.data_array.size, self.n_samples)
        with self.subTest("start"):
            self.assertEqual(data.start, self.ch_ds.start)
        with self.subTest("end"):
            self.assertEqual(data.end, self.ch_ds.end)

    def test_full_slice_too_early(self):
        data = self.ch_ds.time_slice(self.ch_ds.start - 5, n_samples=5096)
        with self.subTest("size"):
            self.assertEqual(data.data_array.size, self.n_samples)
        with self.subTest("start"):
            self.assertEqual(data.start, self.ch_ds.start)
        with self.subTest("end"):
            self.assertEqual(data.end, self.ch_ds.end)

    def test_small_slice_from_end_time(self):
        end = "1980-01-01T00:00:59+00:00"
        data = self.ch_ds.time_slice(self.ch_ds.start, end=end)
        with self.subTest("size"):
            self.assertEqual(data.data_array.size, 60)
        with self.subTest("start"):
            self.assertEqual(data.start, self.ch_ds.start)
        with self.subTest("end"):
            self.assertEqual(data.end, end)

    def test_small_slice_from_n_samples(self):
        n_samples = 60
        data = self.ch_ds.time_slice(self.ch_ds.start, n_samples=n_samples)
        with self.subTest("size"):
            self.assertEqual(data.data_array.size, n_samples)
        with self.subTest("start"):
            self.assertEqual(data.start, self.ch_ds.start)
        with self.subTest("end"):
            self.assertEqual(data.end, "1980-01-01T00:00:59+00:00")

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.m.filename.unlink()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
