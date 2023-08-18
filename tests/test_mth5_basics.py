# -*- coding: utf-8 -*-
"""
Tests for MTh5

Created on Thu Jun 18 16:54:19 2020

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================

import unittest
from pathlib import Path

from mth5.mth5 import MTH5
from mth5 import helpers


fn_path = Path(__file__).parent
# =============================================================================
#
# =============================================================================
helpers.close_open_files()


class TestMTH5Basics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mth5_obj = MTH5()
        self.maxDiff = None

    def test_str(self):
        self.assertEqual(
            self.mth5_obj.__str__(),
            "HDF5 file is closed and cannot be accessed.",
        )

    def test_repr(self):
        self.assertEqual(
            self.mth5_obj.__repr__(),
            "HDF5 file is closed and cannot be accessed.",
        )

    def test_file_type(self):
        self.assertEqual(self.mth5_obj.file_type, "mth5")

    def test_set_file_type_fail(self):
        def set_file_type(value):
            self.mth5_obj.file_type = value

        with self.subTest("bad value"):
            self.assertRaises(ValueError, set_file_type, 10)
        with self.subTest("bad file type"):
            self.assertRaises(ValueError, set_file_type, "asdf")

    def test_set_file_version_fail(self):
        def set_file_version(value):
            self.mth5_obj.file_version = value

        with self.subTest("bad value"):
            self.assertRaises(ValueError, set_file_version, 10)
        with self.subTest("bad file version"):
            self.assertRaises(ValueError, set_file_version, "4")

    def test_set_data_level_fail(self):
        def set_data_level(value):
            self.mth5_obj.data_level = value

        with self.subTest("bad value"):
            self.assertRaises(ValueError, set_data_level, "y")
        with self.subTest("bad data level"):
            self.assertRaises(ValueError, set_data_level, "10")

    def test_filename_fail(self):
        self.mth5_obj.filename = "filename.txt"
        with self.subTest("isinstance path"):
            self.assertIsInstance(self.mth5_obj.filename, Path)
        with self.subTest("extension"):
            self.assertEqual(self.mth5_obj.filename.suffix, ".h5")

    def test_is_read(self):
        self.assertEqual(self.mth5_obj.h5_is_read(), False)

    def test_is_write(self):
        self.assertEqual(self.mth5_obj.h5_is_write(), False)

    def test_validation(self):
        self.assertEqual(self.mth5_obj.validate_file(), False)

    @classmethod
    def tearDownClass(self):
        self.mth5_obj.close_mth5()


class TestWithMTH5(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = Path().cwd().joinpath("test.h5")
        with MTH5() as self.m:
            self.m.open_mth5(self.fn)
            self.m.add_survey("test")

    def test_validate(self):
        self.assertEqual(self.m.validate_file(), False)

    @classmethod
    def tearDownClass(self):
        self.fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
