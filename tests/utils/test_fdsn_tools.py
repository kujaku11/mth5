# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 21:58:30 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from mt_metadata.timeseries import Auxiliary, Electric, Magnetic

from mth5.utils import fdsn_tools


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
            channel_number=3,
            sample_rate=500,
            measurement_azimuth=95,
        )
        self.tx = Auxiliary(
            component="temperature",
            channel_number=6,
            sample_rate=1200,
            measurement_azimuth=105,
            type="temperature",
        )

    def test_location_code(self):
        with self.subTest("ex"):
            self.assertEqual("E1", fdsn_tools.get_location_code(self.ex))

        with self.subTest("hx"):
            self.assertEqual("H3", fdsn_tools.get_location_code(self.hx))

        with self.subTest("temperature"):
            self.assertEqual("T6", fdsn_tools.get_location_code(self.tx))

    def test_period_code(self):
        code = {
            "F": 2000,
            "C": 500,
            "E": 100,
            "B": 50,
            "M": 5,
            "L": 1,
            "V": 0.1,
            "U": 0.01,
            "R": 0.0005,
            "P": 0.00005,
            "T": 0.000005,
            "Q": 0.0000005,
        }

        for key, value in code.items():
            with self.subTest(key):
                self.assertEqual(key, fdsn_tools.get_period_code(value))

    def test_measurment_code(self):
        code = {
            "tilt": "A",
            "creep": "B",
            "calibration": "C",
            "pressure": "D",
            "magnetics": "F",
            "gravity": "G",
            "humidity": "I",
            "temperature": "K",
            "water_current": "O",
            "electric": "Q",
            "rain_fall": "R",
            "linear_strain": "S",
            "tide": "T",
            "wind": "W",
        }

        for key, value in code.items():
            with self.subTest(msg=key):
                self.assertEqual(value, fdsn_tools.get_measurement_code(key))

    def test_orientation_code_horizontal(self):
        code = {0: "N", 5: "N", 20: "1", 90: "E", 55: "2"}

        for key, value in code.items():
            with self.subTest(msg=str(key)):
                self.assertEqual(value, fdsn_tools.get_orientation_code(key))

    def test_orientation_code_vertical(self):
        code = {0: "Z", 5: "Z", 20: "3"}

        for key, value in code.items():
            with self.subTest(msg=str(key)):
                self.assertEqual(
                    value,
                    fdsn_tools.get_orientation_code(key, orientation="vertical"),
                )

    def test_orientation_fail(self):
        self.assertRaises(ValueError, fdsn_tools.get_orientation_code, 0, "juxtapose")

    def test_make_channel_code(self):
        with self.subTest("ex"):
            self.assertEqual("BQN", fdsn_tools.make_channel_code(self.ex))

        with self.subTest("hx"):
            self.assertEqual("CFN", fdsn_tools.make_channel_code(self.hx))

        with self.subTest("temperature"):
            self.assertEqual("FKN", fdsn_tools.make_channel_code(self.tx))

    def test_read_channel_code(self):
        with self.subTest("BQN"):
            self.assertDictEqual(
                {
                    "period": {"min": 10, "max": 80},
                    "component": "electric",
                    "orientation": {"min": 0, "max": 15},
                    "vertical": False,
                },
                fdsn_tools.read_channel_code("BQN"),
            )

        with self.subTest("FQZ"):
            self.assertDictEqual(
                {
                    "period": {"min": 1000, "max": 5000},
                    "component": "electric",
                    "orientation": {"min": 0, "max": 15},
                    "vertical": True,
                },
                fdsn_tools.read_channel_code("FQZ"),
            )

    def test_make_mt_channel(self):
        with self.subTest("ex"):
            self.assertEqual(
                "ex",
                fdsn_tools.make_mt_channel(fdsn_tools.read_channel_code("BQN")),
            )

        with self.subTest("hx"):
            self.assertEqual(
                "hx",
                fdsn_tools.make_mt_channel(fdsn_tools.read_channel_code("FFN")),
            )

        with self.subTest("temperature"):
            self.assertEqual(
                "temperaturez",
                fdsn_tools.make_mt_channel(fdsn_tools.read_channel_code("KKZ")),
            )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
