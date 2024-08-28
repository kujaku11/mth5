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
import numpy as np

from mth5.mth5 import MTH5
from mth5 import helpers
from mth5 import groups
from mth5.mth5 import _default_table_names
from mth5.utils.exceptions import MTH5Error
from mth5.timeseries import ChannelTS, RunTS
from mth5.groups.standards import summarize_metadata_standards
from mt_metadata.utils.mttime import MTime

fn_path = Path(__file__).parent
# =============================================================================
#
# =============================================================================
helpers.close_open_files()


class TestMTH5(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = MTH5(file_version="0.1.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        self.maxDiff = None

    def test_initial_standards_group_size(self):
        stable = self.mth5_obj.standards_group.summary_table
        self.assertGreater(stable.nrows, 1)

    def test_initial_standards_keys(self):
        stable = self.mth5_obj.standards_group.summary_table
        standards_dict = summarize_metadata_standards()
        standards_keys = sorted(
            [
                ss
                for ss in standards_dict.keys()
                if "mth5" not in ss and "hdf5" not in ss
            ]
        )

        stable_keys = sorted(
            [
                ss.decode()
                for ss in list(stable.array["attribute"])
                if "mth5" not in ss.decode() and "hdf5" not in ss.decode()
            ]
        )

        self.assertListEqual(standards_keys, stable_keys)

    def test_default_group_names(self):
        groups = sorted(self.mth5_obj.survey_group.groups_list)
        defaults = sorted(
            self.mth5_obj._default_subgroup_names
            + _default_table_names()
        )

        self.assertListEqual(defaults, groups)

    def test_filename(self):
        self.assertIsInstance(self.mth5_obj.filename, Path)

    def test_is_read(self):
        self.assertEqual(self.mth5_obj.h5_is_read(), True)

    def test_is_write(self):
        self.assertEqual(self.mth5_obj.h5_is_write(), True)

    def test_validation(self):
        self.assertEqual(self.mth5_obj.validate_file(), True)

    def test_set_survey_metadata_attr(self):
        """ Test that we can change a value in the survey metadata and this is written to mth5"""
        survey_metadata = self.mth5_obj.survey_group.metadata
        assert survey_metadata.id is None
        new_survey_id = "MT Survey"
        survey_metadata.id = new_survey_id
        assert survey_metadata.id == new_survey_id
        self.mth5_obj.survey_group.update_metadata(survey_metadata.to_dict())
        assert self.mth5_obj.survey_group.metadata.id == new_survey_id

    def test_add_station(self):
        new_station = self.mth5_obj.add_station("MT001")
        with self.subTest("has_read_metadata"):
            self.assertEqual(True, new_station._has_read_metadata)
        with self.subTest("groups list"):
            self.assertIn("MT001", self.mth5_obj.stations_group.groups_list)
        with self.subTest("isinstance StationGroup"):
            self.assertIsInstance(new_station, groups.StationGroup)
        with self.subTest("get station"):
            sg = self.mth5_obj.get_station("MT001")
            self.assertIsInstance(sg, groups.StationGroup)

        with self.subTest("get station read meatadata"):
            self.assertEqual(True, sg._has_read_metadata)

        with self.subTest("get_station check survey metadata"):
            self.assertListEqual(
                new_station.survey_metadata.station_names, ["MT001"]
            )

    def test_remove_station(self):
        self.mth5_obj.add_station("MT002")
        self.mth5_obj.remove_station("MT002")
        self.assertNotIn("MT002", self.mth5_obj.stations_group.groups_list)

    def test_get_station_fail(self):
        self.assertRaises(MTH5Error, self.mth5_obj.get_station, "MT010")

    def test_add_run(self):
        new_station = self.mth5_obj.add_station("MT003")
        new_run = new_station.add_run("MT003a")
        with self.subTest("has_read_metadata"):
            self.assertEqual(True, new_run._has_read_metadata)
        with self.subTest("groups list"):
            self.assertIn("MT003a", new_station.groups_list)
        with self.subTest("isinstance RunGroup"):
            self.assertIsInstance(new_run, groups.RunGroup)
        with self.subTest("get run"):
            rg = self.mth5_obj.get_run("MT003", "MT003a")
            self.assertIsInstance(rg, groups.RunGroup)

        with self.subTest("check station metadata"):
            self.assertListEqual(new_run.station_metadata.run_list, ["MT003a"])
        with self.subTest("check survey metadata"):
            self.assertListEqual(
                new_run.survey_metadata.stations[0].run_list, ["MT003a"]
            )

    def test_remove_run(self):
        new_station = self.mth5_obj.add_station("MT004")
        new_station.add_run("MT004a")
        new_station.remove_run("MT004a")
        self.assertNotIn("MT004a", new_station.groups_list)

    def test_get_run_fail(self):
        self.assertRaises(MTH5Error, self.mth5_obj.get_run, "MT001", "MT002a")

    def test_add_channel(self):
        new_station = self.mth5_obj.add_station("MT005")
        new_run = new_station.add_run("MT005a")
        new_channel = new_run.add_channel("Ex", "electric", None, shape=(4096,))
        with self.subTest("groups list"):
            self.assertIn("ex", new_run.groups_list)
        with self.subTest("isinstance ElectricDataset"):
            self.assertIsInstance(new_channel, groups.ElectricDataset)
        with self.subTest("get channel"):
            ch = self.mth5_obj.get_channel("MT005", "MT005a", "ex")
            self.assertIsInstance(ch, groups.ElectricDataset)

        with self.subTest("check run metadata"):
            self.assertListEqual(
                new_channel.run_metadata.channels_recorded_all, ["ex"]
            )
        with self.subTest("check station metadata"):
            self.assertListEqual(
                new_channel.station_metadata.runs[
                    "MT005a"
                ].channels_recorded_all,
                ["ex"],
            )
        with self.subTest("check survey metadata"):
            self.assertListEqual(
                new_channel.survey_metadata.stations["MT005"]
                .runs["MT005a"]
                .channels_recorded_all,
                ["ex"],
            )
        with self.subTest("check shape"):
            self.assertTupleEqual(new_channel.hdf5_dataset.shape, (4096,))

    def test_remove_channel(self):
        new_station = self.mth5_obj.add_station("MT006")
        new_run = new_station.add_run("MT006a")
        new_run.add_channel("Ex", "electric", None)
        new_run.remove_channel("Ex")
        self.assertNotIn("ex", new_run.groups_list)

    def test_get_channel_fail(self):
        new_station = self.mth5_obj.add_station("MT007")
        new_station.add_run("MT007a")
        self.assertRaises(
            MTH5Error, self.mth5_obj.get_channel, "MT007", "MT007a", "Ey"
        )

    def test_channel_mtts(self):
        meta_dict = {
            "electric": {
                "component": "Ex",
                "dipole_length": 49.0,
                "measurement_azimuth": 12.0,
                "type": "electric",
                "units": "millivolts",
                "time_period.start": "2020-01-01T12:00:00",
                "sample_rate": 1,
            }
        }
        channel_ts = ChannelTS(
            channel_type="electric",
            data=np.random.rand(4096),
            channel_metadata=meta_dict,
        )

        station = self.mth5_obj.add_station("MT002")
        run = station.add_run("MT002a")
        ex = run.add_channel("Ex", "electric", None)
        ex.from_channel_ts(channel_ts)
        new_ts = ex.to_channel_ts()

        with self.subTest("start"):
            self.assertEqual(channel_ts.start, new_ts.start)
        with self.subTest("time"):
            self.assertTrue(
                channel_ts.data_array.time.to_dict()
                == new_ts.data_array.time.to_dict()
            )

    def test_make_survey_path(self):
        self.assertEqual("/Survey", self.mth5_obj._make_h5_path())

    def test_make_station_path(self):
        self.assertEqual(
            "/Survey/Stations/mt_001",
            self.mth5_obj._make_h5_path(station="mt 001"),
        )

    def test_make_run_path(self):
        self.assertEqual(
            "/Survey/Stations/mt_001/a_001",
            self.mth5_obj._make_h5_path(station="mt 001", run="a 001"),
        )

    def test_make_channel_path(self):
        self.assertEqual(
            "/Survey/Stations/mt_001/a_001/ex",
            self.mth5_obj._make_h5_path(
                station="mt 001", run="a 001", channel="ex"
            ),
        )

    @classmethod
    def tearDownClass(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()


class TestMTH5AddData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = MTH5(file_version="0.1.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        self.maxDiff = None

        ts_list = []
        for comp in ["ex", "ey", "hx", "hy", "hz"]:
            if comp[0] in ["e"]:
                ch_type = "electric"
            elif comp[1] in ["h", "b"]:
                ch_type = "magnetic"
            else:
                ch_type = "auxiliary"
            meta_dict = {
                ch_type: {
                    "component": comp,
                    "dipole_length": 49.0,
                    "measurement_azimuth": 12.0,
                    "type": ch_type,
                    "units": "counts",
                    "time_period.start": "2020-01-01T12:00:00",
                    "sample_rate": 1,
                }
            }
            channel_ts = ChannelTS(
                ch_type, data=np.random.rand(4096), channel_metadata=meta_dict
            )
            ts_list.append(channel_ts)
        run_ts = RunTS(ts_list, {"run": {"id": "MT009a"}})

        self.station = self.mth5_obj.add_station("MT009", survey="test")
        self.rg = self.station.add_run("MT009a")
        self.channel_groups = self.rg.from_runts(run_ts)

    def test_channels(self):
        self.assertListEqual(
            ["ex", "ey", "hx", "hy", "hz"], self.rg.groups_list
        )

        # check to make sure the metadata was transfered
        for cg in self.channel_groups:
            with self.subTest(f"{cg.metadata.component}.start"):
                self.assertEqual(MTime("2020-01-01T12:00:00"), cg.start)
            with self.subTest(f"{cg.metadata.component}.sample_rate"):
                self.assertEqual(1, cg.sample_rate)
            with self.subTest(f"{cg.metadata.component}.n_samples"):
                self.assertEqual(4096, cg.n_samples)

    def test_slice(self):
        r_slice = self.rg.to_runts(start="2020-01-01T12:00:00", n_samples=256)

        with self.subTest("end time"):
            self.assertEqual(r_slice.end, "2020-01-01T12:04:15+00:00")
        with self.subTest("number of samples"):
            self.assertEqual(256, r_slice.dataset.coords.indexes["time"].size)

    def test_station_in_survey_metadata(self):
        self.assertListEqual(
            ["MT009"], self.mth5_obj.survey_group.metadata.station_names
        )

    def test_run_in_station_metadata(self):
        self.assertListEqual(
            ["MT009a"],
            self.mth5_obj.survey_group.metadata.stations[0].run_list,
        )

    def test_channel_in_run_metadata(self):
        self.assertListEqual(
            ["ex", "ey", "hx", "hy", "hz"],
            self.mth5_obj.survey_group.metadata.stations[0]
            .runs[0]
            .channels_recorded_all,
        )

    @classmethod
    def tearDownClass(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()


class TestMTH5GetMethods(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = MTH5(file_version="0.1.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        self.maxDiff = None

        self.station_group = self.mth5_obj.add_station("mt01")
        self.station_group.metadata.location.latitude = 40
        self.station_group.metadata.location.longitude = -120

        self.run_group = self.station_group.add_run("a")
        self.run_group.metadata.time_period.start = "2020-01-01T00:00:00"
        self.run_group.metadata.time_period.end = "2020-01-01T12:00:00"

        self.channel_dataset = self.run_group.add_channel(
            "ex", "electric", None
        )
        self.channel_dataset.metadata.time_period.start = "2020-01-01T00:00:00"
        self.channel_dataset.metadata.time_period.end = "2020-01-01T12:00:00"
        self.channel_dataset.write_metadata()

        self.run_group.update_metadata()
        self.station_group.update_metadata()

    def test_get_station_mth5(self):
        sg = self.mth5_obj.get_station("mt01")

        with self.subTest("has read metadata"):
            self.assertEqual(True, sg._has_read_metadata)

        og_dict = self.station_group.metadata.to_dict(single=True)
        get_dict = sg.metadata.to_dict(single=True)
        for key, original in og_dict.items():
            if "hdf5_reference" != key:
                with self.subTest(key):
                    self.assertEqual(original, get_dict[key])

    def test_get_station_from_stations_group(self):
        sg = self.mth5_obj.survey_group.stations_group.get_station("mt01")

        with self.subTest("has read metadata"):
            self.assertEqual(True, sg._has_read_metadata)

        og_dict = self.station_group.metadata.to_dict(single=True)
        get_dict = sg.metadata.to_dict(single=True)
        for key, original in og_dict.items():
            if "hdf5_reference" != key:
                with self.subTest(key):
                    self.assertEqual(original, get_dict[key])

    def test_get_run_mth5(self):
        rg = self.mth5_obj.get_run("mt01", "a")

        with self.subTest("has read metadata"):
            self.assertEqual(True, rg._has_read_metadata)

        og_dict = self.run_group.metadata.to_dict(single=True)
        get_dict = rg.metadata.to_dict(single=True)
        for key, original in og_dict.items():
            if "hdf5_reference" != key:
                with self.subTest(key):
                    self.assertEqual(original, get_dict[key])

    def test_get_run_from_stations_group(self):
        sg = self.mth5_obj.survey_group.stations_group.get_station("mt01")
        rg = sg.get_run("a")

        with self.subTest("has read metadata"):
            self.assertEqual(True, sg._has_read_metadata)

        og_dict = self.run_group.metadata.to_dict(single=True)
        get_dict = rg.metadata.to_dict(single=True)
        for key, original in og_dict.items():
            if "hdf5_reference" != key:
                with self.subTest(key):
                    self.assertEqual(original, get_dict[key])

    def test_deprecation_update_survey_metadata(self):
        self.assertRaises(
            DeprecationWarning,
            self.mth5_obj.survey_group.update_survey_metadata,
        )

    def test_deprecation_update_station_metadata(self):
        self.assertRaises(
            DeprecationWarning, self.station_group.update_station_metadata
        )

    def test_deprecation_update_run_metadata(self):
        self.assertRaises(
            DeprecationWarning, self.run_group.update_run_metadata
        )

    @classmethod
    def tearDownClass(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()


class TestMTH5InReadMode(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = MTH5(file_version="0.1.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        self.maxDiff = None

        station_group = self.mth5_obj.add_station("mt01")
        station_group.metadata.location.latitude = 40
        station_group.metadata.location.longitude = -120
        station_group.update_metadata()
        self.mth5_obj.close_mth5()

    def test_get_station(self):
        m = MTH5()
        m.open_mth5(self.fn, mode="r")

        sg = m.get_station("mt01")

        sg.metadata.location.latitude = 50
        sg.write_metadata()

        sg = m.get_station("mt01")

        self.assertEqual(sg.metadata.location.latitude, 40)

    @classmethod
    def tearDownClass(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
