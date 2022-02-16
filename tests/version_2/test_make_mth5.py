# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:46:55 2021

@author: jpeacock
"""
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from mth5.clients.make_mth5 import MakeMTH5
from mth5.clients.fdsn import FDSN

# =============================================================================
# Test various inputs for getting metadata
# =============================================================================


class TestMakeMTH5(unittest.TestCase):
    """
    test a csv input to get metadata from IRIS

    """

    def setUp(self):

        self.channels = ["LFE", "LFN", "LFZ", "LQE", "LQN"]
        self.stations = ["CAY10", "CAS04", "NVR08"]
        CAY10 = ["EM","CAY10", '2019-10-07T00:00:00', '2019-10-30T00:00:00']
        CAS04 = ["ZU", "CAS04",  '2020-06-02T19:00:00', '2020-07-13T19:00:00']
        NVR08 = ["ZU", "NVR08", '2020-06-02T19:00:00', '2020-07-13T19:00:00']
        
        request_list = []
        for entry in [CAY10, CAS04, NVR08]:
            for channel in self.channels:
                request_list.append(
                    [entry[0], entry[1], "", channel, entry[2], entry[3]]
                )

        self.csv_fn = Path().cwd().joinpath("test_inventory.csv")
        self.mth5_path = Path().cwd()

        self.make_mth5 = MakeMTH5(mth5_version="0.2.0")
        self.make_mth5.save_path = self.mth5_path
        self.make_mth5.interact = True
        self.fdsn_object = FDSN()
       
        # Turn list into dataframe
        self.metadata_df = pd.DataFrame(
            request_list, columns=self.fdsn_object.column_names
        )
        self.metadata_df.to_csv(self.csv_fn, index=False)

        self.metadata_df_fail = pd.DataFrame(
            request_list, columns=["net", "sta", "loc", "chn", "startdate", "enddate"]
        )

    def test_df_input_inventory(self):
        inv, streams = self.fdsn_object.get_inventory_from_df(
            self.metadata_df, data=False
        )
        with self.subTest(name="stations"):
            self.assertListEqual(
                [self.stations[0]], [ss.code for ss in inv.networks[0].stations]
            )
            
        with self.subTest(name="stations"):
            self.assertListEqual(
                self.stations[1:], [ss.code for ss in inv.networks[1].stations]
            )
            
        with self.subTest(name="channels_CAY10"):
            self.assertListEqual(
                self.channels, [ss.code for ss in inv.networks[1].stations[0].channels]
            )
        with self.subTest(name="channels_CAS04"):
            self.assertListEqual(
                self.channels, [ss.code for ss in inv.networks[1].stations[0].channels]
            )
        with self.subTest(name="channels_NVR08"):
            self.assertListEqual(
                self.channels, [ss.code for ss in inv.networks[1].stations[1].channels]
            )

    def test_csv_input_inventory(self):
        inv, streams = self.fdsn_object.get_inventory_from_df(self.csv_fn, data=False)
        with self.subTest(name="stations"):
            self.assertListEqual(
                [self.stations[0]], [ss.code for ss in inv.networks[0].stations]
            )
            
        with self.subTest(name="stations"):
            self.assertListEqual(
                self.stations[1:], [ss.code for ss in inv.networks[1].stations]
            )
            
        with self.subTest(name="channels_CAY10"):
            self.assertListEqual(
                self.channels, [ss.code for ss in inv.networks[1].stations[0].channels]
            )
        with self.subTest(name="channels_CAS04"):
            self.assertListEqual(
                self.channels, [ss.code for ss in inv.networks[1].stations[0].channels]
            )
        with self.subTest(name="channels_NVR08"):
            self.assertListEqual(
                self.channels, [ss.code for ss in inv.networks[1].stations[1].channels]
            )

    def test_fail_csv_inventory(self):
        self.assertRaises(
            ValueError,
            self.fdsn_object.get_inventory_from_df,
            *(self.metadata_df_fail, self.fdsn_object.client, False),
        )

    def test_fail_wrong_input_type(self):
        self.assertRaises(
            ValueError,
            self.fdsn_object.get_inventory_from_df,
            *(("bad tuple", "bad_tuple"), self.fdsn_object.client, False),
        )

    def test_fail_non_existing_file(self):
        self.assertRaises(
            IOError,
            self.fdsn_object.get_inventory_from_df,
            *("c:\bad\file\name", self.fdsn_object.client, False),
        )

    def test_make_mth5(self):
        self.m = self.make_mth5.from_fdsn_client(self.metadata_df)

        sg = self.m.get_survey("CONUS_South")
        with self.subTest(name="stations"):
            self.assertListEqual(self.stations[1:], sg.stations_group.groups_list)
            
        sg = self.m.get_survey("Kansas_2017_Long_Period")
        with self.subTest(name="stations"):
            self.assertListEqual([self.stations[0]], sg.stations_group.groups_list)

        with self.subTest(name="CAS04_runs"):
            self.assertListEqual(
                ["a", "b", "c", "d"], self.m.get_station("CAS04", "CONUS_South").groups_list
            )

        for run in ["a", "b", "c", "d"]:
            for ch in ["ex", "ey", "hx", "hy", "hz"]:
                with self.subTest(name=f"has data CAS04.{run}.{ch}"):
                    x = self.m.get_channel("CAS04", run, ch, "CONUS_South")
                    self.assertFalse(np.all(x.hdf5_dataset == 0))

        for run in ["a", "b", "c"]:
            for ch in ["ex", "ey", "hx", "hy", "hz"]:
                with self.subTest(name=f"has data NVR08.{run}.{ch}"):
                    x = self.m.get_channel("NVR08", run, ch, "CONUS_South")
                    self.assertFalse(np.all(x.hdf5_dataset == 0))

        self.m.close_mth5()
        # self.m.filename.unlink()

    def tearDown(self):
        self.csv_fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
