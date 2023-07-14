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
from mth5.utils.exceptions import MTH5Error
from mth5.timeseries import ChannelTS, RunTS
from mth5.groups.standards import summarize_metadata_standards
from mt_metadata.utils.mttime import MTime

fn_path = Path(__file__).parent
# =============================================================================
#
# =============================================================================
helpers.close_open_files()

# for some reason this dosen't work when using @classmethod def setUpClass
# keep getting an attribute error in Channel, at least on Git Actions.
class TestMTH5(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = MTH5(file_version="0.2.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        self.survey_group = self.mth5_obj.add_survey("test")
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
        groups = sorted(self.mth5_obj.experiment_group.groups_list)
        defaults = sorted(
            self.mth5_obj._default_subgroup_names
            + ["channel_summary", "tf_summary"]
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

    def test_add_station(self):
        new_station = self.mth5_obj.add_station("MT001", survey="test")
        with self.subTest(name="station exists"):
            self.assertIn("MT001", self.survey_group.stations_group.groups_list)
        with self.subTest(name="is station group"):
            self.assertIsInstance(new_station, groups.StationGroup)
        with self.subTest("get channel"):
            sg = self.mth5_obj.get_station("MT001", survey="test")
            self.assertIsInstance(sg, groups.StationGroup)

    def test_remove_station(self):
        self.mth5_obj.add_station("MT002", survey="test")
        self.mth5_obj.remove_station("MT002", survey="test")
        self.assertNotIn("MT002", self.survey_group.stations_group.groups_list)

    def test_get_station_fail(self):
        self.assertRaises(MTH5Error, self.mth5_obj.get_station, "MT020", "test")

    def test_add_run(self):
        new_station = self.mth5_obj.add_station("MT003", survey="test")
        new_run = new_station.add_run("MT003a")
        with self.subTest("groups list"):
            self.assertIn("MT003a", new_station.groups_list)
        with self.subTest("isinstance RunGroup"):
            self.assertIsInstance(new_run, groups.RunGroup)
        with self.subTest("get run"):
            rg = self.mth5_obj.get_run("MT003", "MT003a", survey="test")
            self.assertIsInstance(rg, groups.RunGroup)

    def test_remove_run(self):
        new_station = self.mth5_obj.add_station("MT004", survey="test")
        new_station.add_run("MT004a")
        new_station.remove_run("MT004a")
        self.assertNotIn("MT004a", new_station.groups_list)

    def test_get_run_fail(self):
        self.assertRaises(
            MTH5Error, self.mth5_obj.get_run, "MT001", "MT002a", "test"
        )

    def test_add_channel(self):
        new_station = self.mth5_obj.add_station("MT005", survey="test")
        new_run = new_station.add_run("MT005a")
        new_channel = new_run.add_channel("Ex", "electric", None)
        with self.subTest("groups list"):
            self.assertIn("ex", new_run.groups_list)
        with self.subTest("isinstance ElectricDataset"):
            self.assertIsInstance(new_channel, groups.ElectricDataset)
        with self.subTest("get channel"):
            try:
                ch = self.mth5_obj.get_channel("MT005", "MT005a", "ex", "test")
                self.assertIsInstance(ch, groups.ElectricDataset)
            except AttributeError:
                print("test_add_channel.get_channel failed with AttributeError")

    def test_remove_channel(self):
        new_station = self.mth5_obj.add_station("MT006", survey="test")
        new_run = new_station.add_run("MT006a")
        new_channel = new_run.add_channel("Ex", "electric", None)
        new_run.remove_channel("Ex")
        self.assertNotIn("ex", new_run.groups_list)

    def test_get_channel_fail(self):
        new_station = self.mth5_obj.add_station("MT007", survey="test")
        new_station.add_run("MT007a")
        self.assertRaises(
            MTH5Error,
            self.mth5_obj.get_channel,
            "MT007",
            "MT007a",
            "Ey",
            "test",
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

        station = self.mth5_obj.add_station("MT008", survey="test")
        run = station.add_run("MT008a")
        ex = run.add_channel("Ex", "electric", None)
        ex.from_channel_ts(channel_ts)
        new_ts = ex.to_channel_ts()

        with self.subTest(name="start times"):
            self.assertEqual(channel_ts.start, new_ts.start)
        with self.subTest(name="metadata"):
            self.assertDictEqual(
                channel_ts._ts.time.to_dict(), new_ts._ts.time.to_dict()
            )

    def test_from_run_ts(self):
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

        station = self.mth5_obj.add_station("MT009", survey="test")
        run = station.add_run("MT009a")
        channel_groups = run.from_runts(run_ts)

        with self.subTest("channels"):
            self.assertListEqual(
                ["ex", "ey", "hx", "hy", "hz"], run.groups_list
            )

        # check to make sure the metadata was transfered
        for cg in channel_groups:
            with self.subTest(f"{cg.metadata.component}.start"):
                self.assertEqual(MTime("2020-01-01T12:00:00"), cg.start)
            with self.subTest(f"{cg.metadata.component}.sample_rate"):
                self.assertEqual(1, cg.sample_rate)
            with self.subTest(f"{cg.metadata.component}.n_samples"):
                self.assertEqual(4096, cg.n_samples)
        # slicing
        r_slice = run.to_runts(start="2020-01-01T12:00:00", n_samples=256)

        with self.subTest("end time"):
            self.assertEqual(r_slice.end, "2020-01-01T12:04:15+00:00")
        with self.subTest("number of samples"):
            self.assertEqual(256, r_slice.dataset.coords.indexes["time"].size)

    def test_make_survey_path(self):
        self.assertEqual(
            "/Experiment/Surveys/test_01",
            self.mth5_obj._make_h5_path(survey="test 01"),
        )

    def test_make_station_path(self):
        self.assertEqual(
            "/Experiment/Surveys/test_01/Stations/mt_001",
            self.mth5_obj._make_h5_path(survey="test 01", station="mt 001"),
        )

    def test_make_run_path(self):
        self.assertEqual(
            "/Experiment/Surveys/test_01/Stations/mt_001/a_001",
            self.mth5_obj._make_h5_path(
                survey="test 01", station="mt 001", run="a 001"
            ),
        )

    def test_make_channel_path(self):
        self.assertEqual(
            "/Experiment/Surveys/test_01/Stations/mt_001/a_001/ex",
            self.mth5_obj._make_h5_path(
                survey="test 01", station="mt 001", run="a 001", channel="ex"
            ),
        )

    @classmethod
    def tearDownClass(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
