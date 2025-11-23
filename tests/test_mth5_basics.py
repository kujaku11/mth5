# -*- coding: utf-8 -*-
"""
Tests for MTh5

Created on Thu Jun 18 16:54:19 2020

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from platform import platform
from typing import Generator

import pytest

from mth5 import __version__ as mth5_version
from mth5 import helpers
from mth5.mth5 import MTH5


fn_path = Path(__file__).parent
# =============================================================================
# Fixtures
# =============================================================================
helpers.close_open_files()


@pytest.fixture(scope="session")
def mth5_obj():
    """Session-scoped MTH5 object for basic tests."""
    obj = MTH5()
    yield obj
    obj.close_mth5()


# =============================================================================
# Test MTH5 Basics
# =============================================================================


class TestMTH5Basics:
    """Test basic MTH5 functionality without file operations."""

    def test_str(self, mth5_obj):
        assert mth5_obj.__str__() == "HDF5 file is closed and cannot be accessed."

    def test_repr(self, mth5_obj):
        assert mth5_obj.__repr__() == "HDF5 file is closed and cannot be accessed."

    def test_file_type(self, mth5_obj):
        assert mth5_obj.file_type == "mth5"

    def test_set_file_type_fail(self, mth5_obj, subtests):
        with subtests.test("bad value"):
            with pytest.raises(ValueError):
                mth5_obj.file_type = 10
        with subtests.test("bad file type"):
            with pytest.raises(ValueError):
                mth5_obj.file_type = "asdf"

    def test_set_file_version_fail(self, mth5_obj, subtests):
        with subtests.test("bad value"):
            with pytest.raises(ValueError):
                mth5_obj.file_version = 10
        with subtests.test("bad file version"):
            with pytest.raises(ValueError):
                mth5_obj.file_version = "4"

    def test_set_data_level_fail(self, mth5_obj, subtests):
        with subtests.test("bad value"):
            with pytest.raises(ValueError):
                mth5_obj.data_level = "y"
        with subtests.test("bad data level"):
            with pytest.raises(ValueError):
                mth5_obj.data_level = "10"

    def test_filename_fail(self, mth5_obj, subtests):
        mth5_obj.filename = "filename.txt"
        with subtests.test("isinstance path"):
            assert isinstance(mth5_obj.filename, Path)
        with subtests.test("extension"):
            assert mth5_obj.filename.suffix == ".h5"

    def test_set_filename(self, mth5_obj):
        fn = Path("fake/path/filename.h5")
        mth5_obj.filename = fn
        assert mth5_obj.filename == fn

    def test_is_read(self, mth5_obj):
        assert mth5_obj.h5_is_read() == False

    def test_is_write(self, mth5_obj):
        assert mth5_obj.h5_is_write() == False

    def test_validation(self, mth5_obj):
        assert mth5_obj.validate_file() == False

    def test_file_attributes(self, mth5_obj):
        file_attrs = {
            "file.type": "MTH5",
            "file.version": "0.2.0",
            "file.access.platform": platform(),
            "mth5.software.version": mth5_version,
            "mth5.software.name": "mth5",
            "data_level": 1,
        }

        for key, value_og in file_attrs.items():
            assert value_og == mth5_obj.file_attributes[key]

    def test_station_list(self, mth5_obj):
        assert mth5_obj.station_list == []

    def test_make_h5_path(self, mth5_obj, subtests):
        with subtests.test("survey"):
            assert mth5_obj._make_h5_path(survey="test") == "/Experiment/Surveys/test"
        with subtests.test("station"):
            assert (
                mth5_obj._make_h5_path(survey="test", station="mt01")
                == "/Experiment/Surveys/test/Stations/mt01"
            )
        with subtests.test("run"):
            assert (
                mth5_obj._make_h5_path(survey="test", station="mt01", run="001")
                == "/Experiment/Surveys/test/Stations/mt01/001"
            )
        with subtests.test("channel"):
            assert (
                mth5_obj._make_h5_path(
                    survey="test", station="mt01", run="001", channel="ex"
                )
                == "/Experiment/Surveys/test/Stations/mt01/001/ex"
            )
        with subtests.test("tf_id"):
            assert (
                mth5_obj._make_h5_path(
                    survey="test",
                    station="mt01",
                    run="001",
                    channel="ex",
                    tf_id="mt01a",
                )
                == "/Experiment/Surveys/test/Stations/mt01/Transfer_Functions/mt01a"
            )


# =============================================================================
# Test With MTH5 File
# =============================================================================


@pytest.fixture(scope="session")
def mth5_test_file(make_worker_safe_path) -> Generator[tuple[MTH5, Path], None, None]:
    """Session-scoped MTH5 file for testing with file operations."""
    fn = make_worker_safe_path("test.h5", Path(__file__).parent)
    with MTH5() as m:
        m.open_mth5(fn)
        m.add_survey("test")

    m = MTH5()
    m.open_mth5(fn)

    yield m, fn

    m.close_mth5()
    if fn.exists():
        fn.unlink()


class TestWithMTH5:
    """Test MTH5 with file operations."""

    def test_validate(self, mth5_test_file):
        m, fn = mth5_test_file
        assert m.validate_file() == True

    def test_station_list(self, mth5_test_file):
        m, fn = mth5_test_file
        assert m.station_list == []

    def test_other_syntax(self, make_worker_safe_path):
        """Test alternative MTH5 usage syntax."""
        fn = make_worker_safe_path("test_other.h5", Path(__file__).parent)

        with MTH5().open_mth5(fn) as m:
            m.add_survey("test2")

        m = MTH5()
        m.open_mth5(fn)

        # test_validate
        assert m.validate_file() == True

        # test_station_list
        assert m.station_list == []

        m.close_mth5()
        if fn.exists():
            fn.unlink()


# =============================================================================
# Test File Version Stability
# =============================================================================


@pytest.fixture(scope="session")
def version_test_files(
    make_worker_safe_path,
) -> Generator[tuple[Path, Path], None, None]:
    """Session-scoped version test files."""
    fn1 = make_worker_safe_path("test_v010.h5", Path(__file__).parent)
    fn2 = make_worker_safe_path("test_v020.h5", Path(__file__).parent)

    with MTH5(file_version="0.1.0") as m:
        m.open_mth5(fn1)
        m.add_station("test_station")

    with MTH5(file_version="0.2.0") as m:
        m.open_mth5(fn2)
        m.add_survey("test_survey")

    yield fn1, fn2

    if fn1.exists():
        fn1.unlink()
    if fn2.exists():
        fn2.unlink()


class TestFileVersionStability:
    """Test file version stability."""

    def test_v1_stays_v1_when_opened_by_v2_obj(self, version_test_files):
        fn1, fn2 = version_test_files
        m = MTH5(file_version="0.2.0")
        assert m.file_version == "0.2.0"
        m.open_mth5(fn1)
        assert m.file_version == "0.1.0"
        m.close_mth5()
        assert m.file_version == "0.2.0"

    def test_v2_stays_v2_when_opened_by_v1_obj(self, version_test_files):
        fn1, fn2 = version_test_files
        m = MTH5(file_version="0.1.0")
        assert m.file_version == "0.1.0"
        m.open_mth5(fn2)
        assert m.file_version == "0.2.0"
        m.close_mth5()
        assert m.file_version == "0.1.0"

    def test_get_version(self, version_test_files):
        from mth5.utils.helpers import get_version

        fn1, fn2 = version_test_files
        file_version = get_version(fn1)
        assert file_version == "0.1.0"
