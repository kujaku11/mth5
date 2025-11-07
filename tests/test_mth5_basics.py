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
from platform import platform

from mth5 import __version__ as mth5_version
from mth5 import helpers
from mth5.mth5 import MTH5


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

    def test_set_filename(self):
        fn = Path("fake/path/filename.h5")
        self.mth5_obj.filename = fn
        self.assertEqual(self.mth5_obj.filename, fn)

    def test_is_read(self):
        self.assertEqual(self.mth5_obj.h5_is_read(), False)

    def test_is_write(self):
        self.assertEqual(self.mth5_obj.h5_is_write(), False)

    def test_validation(self):
        self.assertEqual(self.mth5_obj.validate_file(), False)

    def test_file_attributes(self):
        file_attrs = {
            "file.type": "MTH5",
            "file.version": "0.2.0",
            "file.access.platform": platform(),
            "mth5.software.version": mth5_version,
            "mth5.software.name": "mth5",
            "data_level": 1,
        }

        for key, value_og in file_attrs.items():
            self.assertEqual(value_og, self.mth5_obj.file_attributes[key])

    def test_station_list(self):
        self.assertListEqual([], self.mth5_obj.station_list)

    def test_make_h5_path(self):
        with self.subTest("survey"):
            self.assertEqual(
                self.mth5_obj._make_h5_path(survey="test"),
                "/Experiment/Surveys/test",
            )
        with self.subTest("station"):
            self.assertEqual(
                self.mth5_obj._make_h5_path(survey="test", station="mt01"),
                "/Experiment/Surveys/test/Stations/mt01",
            )
        with self.subTest("run"):
            self.assertEqual(
                self.mth5_obj._make_h5_path(survey="test", station="mt01", run="001"),
                "/Experiment/Surveys/test/Stations/mt01/001",
            )
        with self.subTest("channel"):
            self.assertEqual(
                self.mth5_obj._make_h5_path(
                    survey="test", station="mt01", run="001", channel="ex"
                ),
                "/Experiment/Surveys/test/Stations/mt01/001/ex",
            )
        with self.subTest("tf_id"):
            self.assertEqual(
                self.mth5_obj._make_h5_path(
                    survey="test",
                    station="mt01",
                    run="001",
                    channel="ex",
                    tf_id="mt01a",
                ),
                "/Experiment/Surveys/test/Stations/mt01/Transfer_Functions/mt01a",
            )

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
        self.m.open_mth5(self.m.filename)

    def test_validate(self):
        self.assertEqual(self.m.validate_file(), True)

    def test_station_list(self):
        self.assertListEqual([], self.m.station_list)

    def test_other_syntax(self):
        with MTH5().open_mth5(self.fn) as m:
            m.add_survey("test2")
        m.open_mth5(self.fn)

        # test_validate
        self.assertEqual(m.validate_file(), True)

        # test_station_list
        self.assertListEqual([], m.station_list)
        m.close_mth5()

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.fn.unlink()


class TestFileVersionStability(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn1 = Path().cwd().joinpath("test_v010.h5")
        self.fn2 = Path().cwd().joinpath("test_v020.h5")
        with MTH5(file_version="0.1.0") as self.m:
            self.m.open_mth5(self.fn1)
            self.m.add_station("test_station")
        with MTH5(file_version="0.2.0") as self.m:
            self.m.open_mth5(self.fn2)
            self.m.add_survey("test_survey")

    def test_v1_stays_v1_when_opened_by_v2_obj(self):
        m = MTH5(file_version="0.2.0")
        assert m.file_version == "0.2.0"
        m.open_mth5(self.fn1)
        assert m.file_version == "0.1.0"
        m.close_mth5()
        assert m.file_version == "0.2.0"

    def test_v2_stays_v2_when_opened_by_v1_obj(self):
        m = MTH5(file_version="0.1.0")
        assert m.file_version == "0.1.0"
        m.open_mth5(self.fn2)
        assert m.file_version == "0.2.0"
        m.close_mth5()
        assert m.file_version == "0.1.0"

    def test_get_version(self):
        from mth5.utils.helpers import get_version

        file_version = get_version(self.fn1)
        assert file_version == "0.1.0"

    @classmethod
    def tearDownClass(self):
        self.fn1.unlink()
        self.fn2.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
