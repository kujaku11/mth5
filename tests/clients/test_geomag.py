# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:33:20 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.clients.geomag import GeomagClient

# =============================================================================


class TestGeomagClient(unittest.TestCase):
    def setUp(self):
        self.client = GeomagClient()

    def test_observatory(self):
        self.client.observatory = "frn"
        self.assertEqual("FRN", self.client.observatory)

    def test_observatory_fail(self):
        def set_observatory(value):
            self.client.observatory = value

        with self.subTest("ValueError"):
            self.assertRaises(ValueError, set_observatory, "ten")

        with self.subTest("TypeError"):
            self.assertRaises(TypeError, set_observatory, 10)

    def test_elements(self):
        self.client.elements = "x"
        self.assertListEqual(["X"], self.client.elements)

    def test_elements_from_string(self):
        self.client.elements = "x,y"
        self.assertListEqual(["X", "Y"], self.client.elements)

    def test_elements_fail_input(self):
        def set_elements(value):
            self.client.elements = value

        with self.subTest("ValueError"):
            self.assertRaises(ValueError, set_elements, "v")

        with self.subTest("ValueError_list"):
            self.assertRaises(ValueError, set_elements, ["x", "v"])

        with self.subTest("TypeError"):
            self.assertRaises(TypeError, set_elements, 10)

        with self.subTest("TypeError_list"):
            self.assertRaises(TypeError, set_elements, ["x", 10])

    def test_sample_period(self):
        self.client.sample_period = "1"
        self.assertEqual(1, self.client.sample_period)

    def test_sample_period_fail(self):
        def set_period(value):
            self.client.sample_period = value

        with self.subTest("ValueError_bad_number"):
            self.assertRaises(ValueError, set_period, "p")

        with self.subTest("TypeError"):
            self.assertRaises(TypeError, set_period, [1])

        with self.subTest("not in list"):
            self.assertRaises(ValueError, set_period, 10)

    def test_start(self):
        self.client.start = "2020-01-01T00:00:00+00:00"
        self.assertEqual(self.client.start, "2020-01-01T00:00:00Z")

    def test_end(self):
        self.client.end = "2020-01-01T00:00:00+00:00"
        self.assertEqual(self.client.end, "2020-01-01T00:00:00Z")

    def test_estimate_chunks(self):
        self.client.start = "2020-01-01T00:00:00+00:00"
        self.client.end = "2020-01-02T12:00:00+00:00"

        self.assertListEqual(
            [("2020-01-01T00:00:00Z", "2020-01-02T12:00:00Z")],
            self.client.get_chunks(),
        )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
