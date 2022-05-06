# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:46:55 2021

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path
import pandas as pd

from mth5.clients.make_mth5 import MakeMTH5
from mth5.clients.fdsn import FDSN
from obspy.clients.fdsn.header import FDSNNoDataException
from mth5.utils.mth5_logger import setup_logger

expected_csv = Path(__file__).parent.joinpath("expected.csv")
expected_df = pd.read_csv(expected_csv)
# =============================================================================
# Test various inputs for getting metadata
# =============================================================================


class TestMakeMTH5(unittest.TestCase):
    """
    test a csv input to get metadata from IRIS

    """

    def setUp(self):

        self.make_mth5 = FDSN(mth5_version="0.1.0")

        channels = ["LFE", "LFN", "LFZ", "LQE", "LQN"]
        CAS04 = ["8P", "CAS04", "2020-06-02T18:00:00", "2020-07-13T19:00:00"]
        NVR08 = ["8P", "NVR08", "2020-06-02T18:00:00", "2020-07-13T19:00:00"]

        request_list = []
        for entry in [CAS04, NVR08]:
            for channel in channels:
                request_list.append(
                    [entry[0], entry[1], "", channel, entry[2], entry[3]]
                )
        self.logger = setup_logger("test_make_mth5_v1")
        self.csv_fn = Path().cwd().joinpath("test_inventory.csv")
        self.mth5_path = Path().cwd()

        self.stations = ["CAS04", "NVR08"]
        self.channels = ["LQE", "LQN", "LFE", "LFN", "LFZ"]

        # Turn list into dataframe
        self.metadata_df = pd.DataFrame(
            request_list, columns=self.make_mth5.column_names
        )

        self.metadata_df.to_csv(self.csv_fn, index=False)

        self.metadata_df_fail = pd.DataFrame(
            request_list, columns=["net", "sta", "loc", "chn", "startdate", "enddate"]
        )

    def test_df_input_inventory(self):
        inv, streams = self.make_mth5.get_inventory_from_df(
            self.metadata_df, data=False
        )
        with self.subTest(name="stations"):
            self.assertListEqual(
                sorted(self.stations),
                sorted([ss.code for ss in inv.networks[0].stations]),
            )
        with self.subTest(name="channels_CAS04"):
            self.assertListEqual(
                sorted(self.channels),
                sorted([ss.code for ss in inv.networks[0].stations[0].channels]),
            )
        with self.subTest(name="channels_NVR08"):
            self.assertListEqual(
                sorted(self.channels),
                sorted([ss.code for ss in inv.networks[0].stations[1].channels]),
            )

    def test_csv_input_inventory(self):
        inv, streams = self.make_mth5.get_inventory_from_df(self.csv_fn, data=False)
        with self.subTest(name="stations"):
            self.assertListEqual(
                sorted(self.stations),
                sorted([ss.code for ss in inv.networks[0].stations]),
            )
        with self.subTest(name="channels_CAS04"):
            self.assertListEqual(
                sorted(self.channels),
                sorted([ss.code for ss in inv.networks[0].stations[0].channels]),
            )
        with self.subTest(name="channels_NVR08"):
            self.assertListEqual(
                sorted(self.channels),
                sorted([ss.code for ss in inv.networks[0].stations[1].channels]),
            )

    def test_fail_csv_inventory(self):
        self.assertRaises(
            ValueError,
            self.make_mth5.get_inventory_from_df,
            *(self.metadata_df_fail, self.make_mth5.client, False),
        )

    def test_fail_wrong_input_type(self):
        self.assertRaises(
            ValueError,
            self.make_mth5.get_inventory_from_df,
            *(("bad tuple", "bad_tuple"), self.make_mth5.client, False),
        )

    def test_fail_non_existing_file(self):
        self.assertRaises(
            IOError,
            self.make_mth5.get_inventory_from_df,
            *("c:\bad\file\name", self.make_mth5.client, False),
        )

    def test_make_mth5(self):
        try:
            self.m = self.make_mth5.make_mth5_from_fdsnclient(
                self.metadata_df, self.mth5_path, interact=True
            )

            with self.subTest(name="stations"):
                self.assertListEqual(self.stations, self.m.station_list)
            with self.subTest(name="CAS04_runs"):
                self.assertListEqual(
                    ["Transfer_Functions", "a", "b", "c", "d"],
                    self.m.get_station("CAS04").groups_list,
                )
            for run in ["a", "b", "c", "d"]:
                for ch in ["ex", "ey", "hx", "hy", "hz"]:
                    with self.subTest(name=f"has data CAS04.{run}.{ch}"):
                        x = self.m.get_channel("CAS04", run, ch)
                        self.assertTrue(abs(x.hdf5_dataset[()].mean()) > 0)
            with self.subTest(name="NVR08_runs"):
                self.assertListEqual(
                    ["Transfer_Functions", "a", "b", "c"],
                    self.m.get_station("NVR08").groups_list,
                )
            for run in ["a", "b", "c"]:
                for ch in ["ex", "ey", "hx", "hy", "hz"]:
                    with self.subTest(name=f"has data NVR08.{run}.{ch}"):
                        x = self.m.get_channel("NVR08", run, ch)
                        self.assertTrue(abs(x.hdf5_dataset[()].mean()) > 0)
            # with self.subTest("channel summary"):

            self.m.close_mth5()
            self.m.filename.unlink()
        except FDSNNoDataException as error:
            self.logger.warning(
                "The requested data could not be found on the FDSN IRIS server, check data availability"
            )
            self.logger.error(error)

            raise Exception(
                "The requested data could not be found on the FDSN IRIS server, check data availability"
            )

    def tearDown(self):
        self.csv_fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
