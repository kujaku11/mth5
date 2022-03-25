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

from mth5 import mth5
from mth5.utils.exceptions import MTH5Error
from mth5.timeseries import ChannelTS, RunTS
from mth5.groups.standards import summarize_metadata_standards
from mt_metadata.utils.mttime import MTime

fn_path = Path(__file__).parent
# =============================================================================
#
# =============================================================================
mth5.helpers.close_open_files()


class TestMTH5(unittest.TestCase):
    def setUp(self):
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = mth5.MTH5(file_version="0.1.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")

    def test_initial_standards_group_size(self):
        stable = self.mth5_obj.standards_group.summary_table
        self.assertGreater(stable.nrows, 1)

    def test_initial_standards_keys(self):
        stable = self.mth5_obj.standards_group.summary_table
        standards_dict = summarize_metadata_standards()
        standards_keys = sorted(list(standards_dict.keys()))

        stable_keys = sorted([ss.decode() for ss in list(stable.array["attribute"])])

        self.assertListEqual(standards_keys, stable_keys)

    def test_default_group_names(self):
        groups = sorted(self.mth5_obj.survey_group.groups_list)
        defaults = sorted(self.mth5_obj._default_subgroup_names + 
                          ["channel_summary", "tf_summary"])

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
        new_station = self.mth5_obj.add_station("MT001")
        self.assertIn("MT001", self.mth5_obj.stations_group.groups_list)
        self.assertIsInstance(new_station, mth5.groups.StationGroup)

    def test_remove_station(self):
        self.mth5_obj.add_station("MT001")
        self.mth5_obj.remove_station("MT001")
        self.assertNotIn("MT001", self.mth5_obj.stations_group.groups_list)

    def test_get_station_fail(self):
        self.assertRaises(MTH5Error, self.mth5_obj.get_station, "MT002")

    def test_add_run(self):
        new_station = self.mth5_obj.add_station("MT001")
        new_run = new_station.add_run("MT001a")
        self.assertIn("MT001a", new_station.groups_list)
        self.assertIsInstance(new_run, mth5.groups.RunGroup)

    def test_remove_run(self):
        new_station = self.mth5_obj.add_station("MT001")
        new_station.add_run("MT001a")
        new_station.remove_run("MT001a")
        self.assertNotIn("MT001a", new_station.groups_list)

    def test_get_run_fail(self):
        self.assertRaises(MTH5Error, self.mth5_obj.get_run, "MT001", "MT002a")

    def test_add_channel(self):
        new_station = self.mth5_obj.add_station("MT001")
        new_run = new_station.add_run("MT001a")
        new_channel = new_run.add_channel("Ex", "electric", None)
        self.assertIn("ex", new_run.groups_list)
        self.assertIsInstance(new_channel, mth5.groups.ElectricDataset)

    def test_remove_channel(self):
        new_station = self.mth5_obj.add_station("MT001")
        new_run = new_station.add_run("MT001a")
        new_channel = new_run.add_channel("Ex", "electric", None)
        new_run.remove_channel("Ex")
        self.assertNotIn("ex", new_run.groups_list)

    def test_get_channel_fail(self):
        new_station = self.mth5_obj.add_station("MT001")
        new_station.add_run("MT001a")
        self.assertRaises(MTH5Error, self.mth5_obj.get_channel, "MT001", "MT001a", "Ey")

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

        self.assertEqual(channel_ts.start, new_ts.start)
        self.assertTrue(channel_ts._ts.time.to_dict() == new_ts._ts.time.to_dict())

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

        run_ts = RunTS(ts_list, {"id": "MT002a"})

        station = self.mth5_obj.add_station("MT002")
        run = station.add_run("MT002a")
        channel_groups = run.from_runts(run_ts)

        self.assertListEqual(["ex", "ey", "hx", "hy", "hz"], run.groups_list)

        # check to make sure the metadata was transfered
        for cg in channel_groups:
            self.assertEqual(MTime("2020-01-01T12:00:00"), cg.start)
            self.assertEqual(1, cg.sample_rate)
            self.assertEqual(4096, cg.n_samples)

        # check the summary table

    def tearDown(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
