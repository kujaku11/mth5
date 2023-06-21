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

# =============================================================================
# Test various inputs for getting metadata
# =============================================================================


class TestMakeMTH5FDSNInventory(unittest.TestCase):
    """
    test a csv input to get metadata from IRIS

    """

    @classmethod
    def setUpClass(self):

        self.fdsn = FDSN(mth5_version="0.2.0")
        self.fdsn.client = "IRIS"
        self.make_mth5 = MakeMTH5(
            mth5_version="0.2.0", interact=True, save_path=Path().cwd()
        )

        channels = ["LFE", "LFN", "LFZ", "LQE", "LQN"]
        CAS04 = ["8P", "CAS04", "2020-06-02T18:00:00", "2020-07-13T19:00:00"]
        NVR08 = ["8P", "NVR08", "2020-06-02T18:00:00", "2020-07-13T19:00:00"]

        request_list = []
        for entry in [CAS04, NVR08]:
            for channel in channels:
                request_list.append(
                    [entry[0], entry[1], "", channel, entry[2], entry[3]]
                )
        self.logger = setup_logger("test_make_mth5_v2")
        self.csv_fn = Path().cwd().joinpath("test_inventory.csv")
        self.mth5_path = Path().cwd()

        self.stations = ["CAS04", "NVR08"]
        self.channels = ["LQE", "LQN", "LFE", "LFN", "LFZ"]

        # Turn list into dataframe
        self.metadata_df = pd.DataFrame(
            request_list, columns=self.fdsn.request_columns
        )
        self.metadata_df.to_csv(self.csv_fn, index=False)

        self.metadata_df_fail = pd.DataFrame(
            request_list,
            columns=["net", "sta", "loc", "chn", "startdate", "enddate"],
        )

    def test_df_input_inventory(self):
        inv, streams = self.fdsn.get_inventory_from_df(
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
                sorted(
                    list(
                        set(
                            [
                                ss.code
                                for ss in inv.networks[0].stations[0].channels
                            ]
                        )
                    )
                ),
            )
        with self.subTest(name="channels_NVR08"):
            self.assertListEqual(
                sorted(self.channels),
                sorted(
                    list(
                        set(
                            [
                                ss.code
                                for ss in inv.networks[0].stations[1].channels
                            ]
                        )
                    )
                ),
            )

    def test_csv_input_inventory(self):
        inv, streams = self.fdsn.get_inventory_from_df(self.csv_fn, data=False)
        with self.subTest(name="stations"):
            self.assertListEqual(
                sorted(self.stations),
                sorted([ss.code for ss in inv.networks[0].stations]),
            )

        with self.subTest(name="channels_CAS04"):
            self.assertListEqual(
                sorted(self.channels),
                sorted(
                    list(
                        set(
                            [
                                ss.code
                                for ss in inv.networks[0].stations[0].channels
                            ]
                        )
                    )
                ),
            )
        with self.subTest(name="channels_NVR08"):
            self.assertListEqual(
                sorted(self.channels),
                sorted(
                    list(
                        set(
                            [
                                ss.code
                                for ss in inv.networks[0].stations[1].channels
                            ]
                        )
                    )
                ),
            )

    def test_fail_csv_inventory(self):
        self.assertRaises(
            ValueError,
            self.fdsn.get_inventory_from_df,
            *(self.metadata_df_fail, self.fdsn.client, False),
        )

    def test_fail_wrong_input_type(self):
        self.assertRaises(
            ValueError,
            self.fdsn.get_inventory_from_df,
            *(("bad tuple", "bad_tuple"), self.fdsn.client, False),
        )

    def test_fail_non_existing_file(self):
        self.assertRaises(
            IOError,
            self.fdsn.get_inventory_from_df,
            *("c:\bad\file\name", self.fdsn.client, False),
        )

    def test_h5_parameters(self):
        with self.subTest("compression"):
            self.assertEqual(self.make_mth5.compression, "gzip")
        with self.subTest("compression_options"):
            self.assertEqual(self.make_mth5.compression_opts, 4)
        with self.subTest("shuffle"):
            self.assertEqual(self.make_mth5.shuffle, True)
        with self.subTest("fletcher32"):
            self.assertEqual(self.make_mth5.fletcher32, True)
        with self.subTest("data_level"):
            self.assertEqual(self.make_mth5.data_level, 1)
        with self.subTest("file_version"):
            self.assertEqual(self.make_mth5.mth5_version, "0.2.0")
        with self.subTest("save_path"):
            self.assertEqual(self.make_mth5.save_path, self.mth5_path)
        with self.subTest("interact"):
            self.assertEqual(self.make_mth5.interact, True)

    def test_fdsn_h5_parameters(self):
        with self.subTest("compression"):
            self.assertEqual(self.fdsn.compression, "gzip")
        with self.subTest("compression_options"):
            self.assertEqual(self.fdsn.compression_opts, 4)
        with self.subTest("shuffle"):
            self.assertEqual(self.fdsn.shuffle, True)
        with self.subTest("fletcher32"):
            self.assertEqual(self.fdsn.fletcher32, True)
        with self.subTest("data_level"):
            self.assertEqual(self.fdsn.data_level, 1)
        with self.subTest("file_version"):
            self.assertEqual(self.fdsn.mth5_version, "0.2.0")

    @classmethod
    def tearDownClass(self):
        self.csv_fn.unlink()


@unittest.skipIf(
    "peacock" not in str(Path().cwd().as_posix()),
    "Test is too long, have to download data from IRIS",
)
class TestMakeMTH5(unittest.TestCase):
    """
    test a csv input to get metadata from IRIS

    """

    @classmethod
    def setUpClass(self):

        self.fdsn = FDSN(mth5_version="0.2.0")
        self.fdsn.client = "IRIS"
        self.make_mth5 = MakeMTH5(
            mth5_version="0.2.0", interact=True, save_path=Path().cwd()
        )

        channels = ["LFE", "LFN", "LFZ", "LQE", "LQN"]
        CAS04 = ["8P", "CAS04", "2020-06-02T18:00:00", "2020-07-13T19:00:00"]
        NVR08 = ["8P", "NVR08", "2020-06-02T18:00:00", "2020-07-13T19:00:00"]

        request_list = []
        for entry in [CAS04, NVR08]:
            for channel in channels:
                request_list.append(
                    [entry[0], entry[1], "", channel, entry[2], entry[3]]
                )
        self.logger = setup_logger("test_make_mth5_v2")
        self.csv_fn = Path().cwd().joinpath("test_inventory.csv")
        self.mth5_path = Path().cwd()

        self.stations = ["CAS04", "NVR08"]
        self.channels = ["LQE", "LQN", "LFE", "LFN", "LFZ"]

        # Turn list into dataframe
        self.metadata_df = pd.DataFrame(
            request_list, columns=self.fdsn.request_columns
        )
        self.metadata_df.to_csv(self.csv_fn, index=False)

        self.metadata_df_fail = pd.DataFrame(
            request_list,
            columns=["net", "sta", "loc", "chn", "startdate", "enddate"],
        )

        try:
            self.m = self.make_mth5.from_fdsn_client(
                self.metadata_df, client="IRIS"
            )

        except FDSNNoDataException as error:
            self.logger.warning(
                "The requested data could not be found on the FDSN IRIS server, check data availability"
            )
            self.logger.error(error)

            raise Exception(
                "The requested data could not be found on the FDSN IRIS server, check data availability"
            )

    def test_survey(self):

        sg = self.m.get_survey("CONUS_South")
        self.assertEqual(sg.metadata.id, "CONUS South")

    def test_stations(self):
        sg = self.m.get_survey("CONUS_South")
        self.assertListEqual(self.stations, sg.stations_group.groups_list)

    def test_cas04_runs_list(self):
        self.assertListEqual(
            [
                "Fourier_Coefficients",
                "Transfer_Functions",
                "a",
                "b",
                "c",
                "d",
            ],
            self.m.get_station("CAS04", "CONUS_South").groups_list,
        )

    def test_cas04_channels(self):
        for run in ["a", "b", "c", "d"]:
            for ch in ["ex", "ey", "hx", "hy", "hz"]:
                x = self.m.get_channel("CAS04", run, ch, "CONUS_South")

                with self.subTest(name=f"has data CAS04.{run}.{ch}"):
                    self.assertTrue(abs(x.hdf5_dataset[()].mean()) > 0)

                with self.subTest(name=f"has metadata CAS04.{run}.{ch}"):
                    self.assertEqual(x.metadata.component, ch)

    def test_cas04_channels_to_ts(self):
        for run in ["a", "b", "c", "d"]:
            for ch in ["ex", "ey", "hx", "hy", "hz"]:
                x = self.m.get_channel(
                    "CAS04", run, ch, "CONUS_South"
                ).to_channel_ts()
                with self.subTest(name=f"has data CAS04.{run}.{ch}"):
                    self.assertTrue(abs(x.ts.mean()) > 0)

                with self.subTest(name=f"has metadata CAS04.{run}.{ch}"):
                    self.assertEqual(x.component, ch)

    def test_nvr08_channels(self):
        for run in ["a", "b", "c"]:
            for ch in ["ex", "ey", "hx", "hy", "hz"]:
                x = self.m.get_channel("NVR08", run, ch, "CONUS_South")
                with self.subTest(name=f"has data NVR08.{run}.{ch}"):
                    self.assertTrue(abs(x.hdf5_dataset[()].mean()) > 0)

                with self.subTest(name="filters"):
                    self.assertTrue(x.metadata.filter.name != [])

    def test_nvr08_channels_to_ts(self):
        for run in ["a", "b", "c"]:
            for ch in ["ex", "ey", "hx", "hy", "hz"]:
                x = self.m.get_channel(
                    "NVR08", run, ch, "CONUS_South"
                ).to_channel_ts()
                with self.subTest(name=f"has data NVR08.{run}.{ch}"):
                    self.assertTrue(abs(x.ts.mean()) > 0)

                with self.subTest(name=f"has metadata NVR08.{run}.{ch}"):
                    self.assertEqual(x.component, ch)

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.m.filename.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
