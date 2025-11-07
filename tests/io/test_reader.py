# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:30:32 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.io import reader


# =============================================================================


class TestReader(unittest.TestCase):
    def test_get_reader_bin(self):
        key, reader_function = reader.get_reader("bin")
        self.assertEqual(key, "nims")

    def test_get_reader_z3d(self):
        key, reader_function = reader.get_reader("z3d")
        self.assertEqual(key, "zen")

    def test_get_reader_asc(self):
        key, reader_function = reader.get_reader("asc")
        self.assertEqual(key, "usgs_ascii")

    def test_get_reader_txt(self):
        key, reader_function = reader.get_reader("txt")
        self.assertEqual(key, "lemi424")

    def test_get_reader_td_150(self):
        key, reader_function = reader.get_reader("td_150")
        self.assertEqual(key, "phoenix")

    def test_get_reader_td_24k(self):
        key, reader_function = reader.get_reader("td_24k")
        self.assertEqual(key, "phoenix")

    def test_get_reader_miniseed(self):
        key, reader_function = reader.get_reader("mseed")
        self.assertEqual(key, "miniseed")

    def test_get_reader_fail(self):
        self.assertRaises(ValueError, reader.get_reader, "dat")

    def test_input_fail(self):
        self.assertRaises(IOError, reader.read_file, "y.txt")

    def test_input_fail_list(self):
        self.assertRaises(IOError, reader.read_file, ["y.txt"])


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
