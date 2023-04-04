# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:30:08 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
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
        self.assertListEqual(
            [1000], get_decimation_sample_rates(1000, 1000, 8)
        )


class TestMakeDtCoordinates(unittest.TestCase):
    def test_input_none(self):
        dt = make_dt_coordinates(None, None, 16, None)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00"))
        with self.subTest("end"):
            self.assertEqual(MTime(dt[-1]), MTime("1980-01-01T00:00:15"))
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_ms(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0010", 1, 16, None)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00.001"))
        with self.subTest("end"):
            self.assertEqual(MTime(dt[-1]), MTime("1980-01-01T00:00:15.001"))
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_us(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0000010", 1, 16, None)

        with self.subTest("start"):
            self.assertEqual(MTime(dt[0]), MTime("1980-01-01T00:00:00.000001"))
        with self.subTest("end"):
            self.assertEqual(
                MTime(dt[-1]), MTime("1980-01-01T00:00:15.000001")
            )
        with self.subTest("length"):
            self.assertEqual(16, len(dt))

    def test_sig_figs_ns(self):
        dt = make_dt_coordinates("1980-01-01T00:00:00.0000000010", 1, 16, None)

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
