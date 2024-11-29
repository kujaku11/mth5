# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:30:08 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import unittest

from mt_metadata.utils.mttime import MTime

from mth5.timeseries.ts_helpers import (
    make_dt_coordinates,
    get_decimation_sample_rates,
    _count_decimal_sig_figs,
)

# =============================================================================


class TestGetDecimationSampleRates(unittest.TestCase):
    def test_4096_to_1(self):
        self.assertListEqual(
            [512, 64, 8, 1], get_decimation_sample_rates(4096, 1, 8)
        )

    def test_1000_to_1(self):
        self.assertListEqual(
            [125, 16, 2, 1], get_decimation_sample_rates(1000, 1, 8)
        )

    def test_1000_to_1000(self):
        self.assertListEqual([1000], get_decimation_sample_rates(1000, 1000, 8))


class TestMakeDtCoordinates(unittest.TestCase):
    def test_input_none(self):
        dt = make_dt_coordinates(None, None, 16)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00"))
        with self.subTest("end"):
            self.assertEqual(MTime(dt[-1]), MTime("1980-01-01T00:00:15"))
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_ms(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0010", 1, 16)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00.001"))
        with self.subTest("end"):
            self.assertEqual(MTime(dt[-1]), MTime("1980-01-01T00:00:15.001"))
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_us(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0000010", 1, 16)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00.000001"))
        with self.subTest("end"):
            self.assertEqual(MTime(dt[-1]), MTime("1980-01-01T00:00:15.000001"))
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_ns(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0000000010", 1, 16)

        with self.subTest("start"):
            self.assertEqual(
                MTime(dt[0]), MTime("1980-01-01T00:00:00.000000001")
            )
        with self.subTest("end"):
            self.assertEqual(
                MTime(dt[-1]), MTime("1980-01-01T00:00:15.000000001")
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sr_sig_figs_ms(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.000", 16, 16)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00.00"))
        with self.subTest("end"):
            self.assertEqual(MTime(dt[-1]), MTime("1980-01-01T00:00:00.9375"))
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sr_sig_figs_us(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.00000", 256, 16)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00.0000"))
        with self.subTest("end"):
            self.assertEqual(MTime(dt[-1]), MTime("1980-01-01T00:00:00.058594"))
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sr_sig_figs_ns(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.00000", 4096, 16)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00.0000"))
        with self.subTest("end"):
            self.assertAlmostEqual(
                MTime(dt[-1]).epoch_seconds,
                MTime("1980-01-01T00:00:00.003662109").epoch_seconds,
                6,
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_fix_issue_263(self):
        """
            Note that passing endtime explicitly vs not can causing different values in time coordinates.


        Returns
        -------

        """
        end_str = "2023-10-14T19:47:31.176479359+00:00"
        start_str = "2023-10-14T19:47:23.978079359+00:00"

        dt = 0.13088
        sr = 1 / dt
        n_samples = 56

        tmp1 = make_dt_coordinates(start_str, sample_rate=sr, n_samples=n_samples, end_time=end_str)
        tmp2 = make_dt_coordinates(start_str, sample_rate=sr, n_samples=n_samples, end_time=None)

        delta_t1 = tmp1.diff()[1:]
        delta_t2 = tmp2.diff()[1:]

        # This assertion indicates that delta_t1 is not uniform, whereas delta_t2 is.
        assert len(delta_t1.unique()) == 2
        assert len(delta_t2.unique()) == 1

        # This assertion indicates that the difference is in the first delta.
        assert delta_t1[0] != delta_t2[0]
        assert (delta_t1[1:] == delta_t2[1:]).all()


class TestDecimalSigFigs(unittest.TestCase):
    def test_sig_figs(self):
        for ii in range(1, 12, 1):
            value = f".{ii:0{ii}}1"
            sig_figs = _count_decimal_sig_figs(value)
            with self.subTest(value):
                self.assertEqual(sig_figs, ii + 1)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
