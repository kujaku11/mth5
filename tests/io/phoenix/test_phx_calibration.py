# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:20:14 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5.io.phoenix.readers import PhoenixCalibration


# =============================================================================
cal_fn = Path(__file__).parent.joinpath("example_rxcal.json")


class TestPHXCalibrations(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.c = PhoenixCalibration(cal_fn)

    def test_has_channel(self):
        for ch in ["e1", "e2", "h1", "h2", "h3"]:
            with self.subTest(ch):
                self.assertTrue(hasattr(self.c, ch))

    def test_has_lp(self):
        for ch in ["e1", "e2", "h1", "h2", "h3"]:
            for lp in [10000, 1000, 100, 10]:
                with self.subTest(f"{ch}_lp{lp}"):
                    self.assertIn(lp, getattr(self.c, ch).keys())

    def test_get_filter(self):
        for ch in ["e1", "e2", "h1", "h2", "h3"]:
            for lp in [10000, 1000, 100, 10]:
                fap = self.c.get_filter(ch, lp)
                with self.subTest(f"{ch}_{lp}_units_in"):
                    self.assertEqual(fap.units_in, "Volt")
                with self.subTest(f"{ch}_{lp}_units_out"):
                    self.assertEqual(fap.units_out, "Volt")

                with self.subTest(f"{ch}_{lp}_name"):
                    self.assertEqual(
                        fap.name,
                        f"{self.c.base_filter_name}_{ch}_{lp}hz_lowpass",
                    )

                with self.subTest(f"{ch}_{lp}_calibration_date"):
                    self.assertEqual(self.c.calibration_date, fap.calibration_date)

                with self.subTest(f"{ch}_{lp}_frequencies"):
                    self.assertTrue((fap.frequencies != 0).all())

                with self.subTest(f"{ch}_{lp}_amplitudes"):
                    self.assertTrue((fap.amplitudes != 0).all())

                with self.subTest(f"{ch}_{lp}_phases"):
                    self.assertTrue((fap.phases != 0).all())

                with self.subTest(f"{ch}_{lp}_max_frequency"):
                    self.assertTrue(lp == self.c.get_max_freq(fap.frequencies))

    def test_get_filter_fail_key_error(self):
        self.assertRaises(KeyError, self.c.get_filter, "e1", 2)

    def test_get_filter_fail_attribute_error(self):
        self.assertRaises(AttributeError, self.c.get_filter, "bx", 100)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
