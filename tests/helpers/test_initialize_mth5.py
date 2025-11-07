# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:56:09 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5.helpers import close_open_files
from mth5.mth5 import MTH5
from mth5.utils.helpers import initialize_mth5


# =============================================================================


class TestInitializeMTH5(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.wd = Path().cwd()
        self.m = MTH5()
        self.m.open_mth5(self.wd.joinpath("test.h5"), mode="a")

    def test_has_open_file(self):
        self.m = initialize_mth5(self.wd.joinpath("test.h5"), "w")
        self.assertIsInstance(self.m, MTH5)

    @classmethod
    def tearDownClass(self):
        close_open_files()
        self.wd.joinpath("test.h5").unlink()


class TestInitializeMTH502(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.wd = Path().cwd()
        self.m = MTH5()
        self.m.open_mth5(self.wd.joinpath("test.h5"), mode="a")

    def test_has_open_file(self):
        m = initialize_mth5(self.wd.joinpath("test.h5"), "a")
        self.assertIsInstance(m, MTH5)

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.wd.joinpath("test.h5").unlink()


class TestCompression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.wd = Path().cwd()

    def test_has_open_file(self):
        self.m = initialize_mth5(self.wd.joinpath("test.h5"), "w")
        self.assertIsInstance(self.m, MTH5)

    def test_validate_compression(self):
        """COMPRESSION = ["lzf", "gzip", "szip", None]"""
        from mth5.helpers import COMPRESSION, COMPRESSION_LEVELS

        for compression_type in COMPRESSION:
            for compression_level in COMPRESSION_LEVELS[compression_type]:
                self.m = MTH5(
                    compression=compression_type, compression_opts=compression_level
                )
                self.m.open_mth5(self.wd.joinpath("test.h5"), mode="a")
                self.m = initialize_mth5(self.wd.joinpath("test.h5"), "w")
                self.assertIsInstance(self.m, MTH5)
                self.m.close_mth5()

    @classmethod
    def tearDownClass(self):
        close_open_files()
        self.wd.joinpath("test.h5").unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
