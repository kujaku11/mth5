# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:07:30 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest

import numpy as np
import xarray as xr

from mth5.timeseries.channel_ts import make_dt_coordinates


# =============================================================================


class TestChannelScipyFilters(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.n_samples = 4096
        self.t = np.arange(self.n_samples)
        self.data = (
            np.sum(
                [
                    np.cos(2 * np.pi * w * self.t + phi)
                    for w, phi in zip(
                        np.logspace(-3, 3, 20), np.random.rand(20)
                    )
                ],
                axis=0,
            )
            + 1.5 * self.t
            + 0.5
        )

        self.sample_rate = 64

        dt_index = make_dt_coordinates(
            "2020-01-01T00:00:00", self.sample_rate, self.n_samples
        )

        self.ch = xr.DataArray(self.data, coords={"time": dt_index}, name="ex")

    def test_decimate(self):
        decimated_ch = self.ch.sps_filters.decimate(1)

        with self.subTest("sample rate"):
            self.assertEqual(decimated_ch.sps_filters.fs, 1.0)
        with self.subTest("start"):
            self.assertEqual(
                self.ch.coords["time"][0].values,
                decimated_ch.coords["time"][0].values,
            )
        with self.subTest("end"):
            self.assertNotEqual(
                self.ch.coords["time"][-1].values,
                decimated_ch.coords["time"][-1].values,
            )
        with self.subTest("size"):
            self.assertEqual(64, decimated_ch.sizes["time"])

    def test_dtrend(self):
        dtrend_ch = self.ch.sps_filters.detrend()

        self.assertEqual(0, round(dtrend_ch.data.mean(), 7))

    def test_dt(self):
        self.assertEqual(1 / 64, self.ch.sps_filters.dt)

    def test_fs(self):
        self.assertEqual(self.sample_rate, self.ch.sps_filters.fs)

    def test_dx(self):
        self.assertListEqual(
            [1 / self.sample_rate], self.ch.sps_filters.dx.tolist()
        )

    def test_resample_poly_odd_ratio(self):
        a = self.ch.sps_filters.resample_poly(3)
        with self.subTest("number of samples"):
            self.assertEqual(a.size, self.n_samples / (self.sample_rate / 3))
        with self.subTest("start time"):
            self.assertTrue(self.ch.time[0].data == a.time[0].data)
        with self.subTest("end time"):
            self.assertTrue(self.ch.time[-1].data == a.time[-1].data)
        with self.subTest("sample rate"):
            self.assertNotEqual(a.sps_filters.fs, 3)

    def test_resample_poly_even_ratio(self):
        a = self.ch.sps_filters.resample_poly(4)
        with self.subTest("number of samples"):
            self.assertEqual(a.size, self.n_samples / (self.sample_rate / 4))
        with self.subTest("start time"):
            self.assertTrue(self.ch.time[0].data == a.time[0].data)
        with self.subTest("end time"):
            self.assertTrue(self.ch.time[-16].data == a.time[-1].data)
        with self.subTest("sample rate"):
            self.assertEqual(a.sps_filters.fs, 4)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
