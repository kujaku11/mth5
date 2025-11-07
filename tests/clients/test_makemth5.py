# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:10:51 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5.clients import MakeMTH5


# =============================================================================


class TestMakeMTH5v1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = MakeMTH5(mth5_version="0.1.0", interact=True, save_path=None)

    def test_mth5_version(self):
        self.assertEqual(self.m.mth5_version, "0.1.0")

    def test_interact_true(self):
        self.assertEqual(self.m.interact, True)

    def test_save_path(self):
        self.assertEqual(self.m.save_path, Path().cwd())

    def test_compression(self):
        self.assertEqual(self.m.h5_compression, "gzip")

    def test_compression_opts(self):
        self.assertEqual(self.m.h5_compression_opts, 4)

    def test_shuffle(self):
        self.assertEqual(self.m.h5_shuffle, True)

    def test_fletcher32(self):
        self.assertEqual(self.m.h5_fletcher32, True)

    def test_data_level(self):
        self.assertEqual(self.m.h5_data_level, 1)


class TestMakeMTH5v2(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m = MakeMTH5(
            mth5_version="0.2.0",
            interact=False,
            save_path=None,
            h5_compression=None,
            h5_compression_opts=None,
            h5_shuffle=False,
            h5_fletcher32=False,
            h5_data_level=2,
        )

    def test_mth5_version(self):
        self.assertEqual(self.m.mth5_version, "0.2.0")

    def test_interact_true(self):
        self.assertEqual(self.m.interact, False)

    def test_save_path(self):
        self.assertEqual(self.m.save_path, Path().cwd())

    def test_compression(self):
        self.assertEqual(self.m.h5_compression, None)

    def test_compression_opts(self):
        self.assertEqual(self.m.h5_compression_opts, None)

    def test_shuffle(self):
        self.assertEqual(self.m.h5_shuffle, False)

    def test_fletcher32(self):
        self.assertEqual(self.m.h5_fletcher32, False)

    def test_data_level(self):
        self.assertEqual(self.m.h5_data_level, 2)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
