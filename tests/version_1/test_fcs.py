# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:59:26 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
# import pytest
import unittest

import xarray
import xarray as xr

from mth5.mth5 import MTH5
from mth5.utils.fc_tools import make_multistation_spectrogram
from mth5.utils.fc_tools import FCRunChunk
from mth5.utils.fc_tools import MultivariateDataset

from mt_metadata.utils.mttime import MTime

# =============================================================================
fn_path = Path(__file__).parent
csv_fn = fn_path.joinpath("test1_dec_level_3.csv")
h5_filename = fn_path.joinpath("fc_test.h5")


#@pytest.fixture
def create_mth5_with_some_test_data():
    """

    Returns
    -------

    """
    # get some test data to pack into mth5
    ds = read_fc_csv(csv_fn)

    m = MTH5()
    m.file_version = "0.1.0"
    m.open_mth5(h5_filename)

    # Add a station
    station_group = m.add_station("mt01")
    fc_group = (
        station_group.fourier_coefficients_group.add_fc_group(
            "processing_run_01"
        )
    )

    decimation_level = fc_group.add_decimation_level("3")
    expected_sr_decimation_level = 0.015380859375
    decimation_level.from_xarray(ds, expected_sr_decimation_level)
    decimation_level.update_metadata()
    fc_group.update_metadata()

    # Add a second station (same as first but scaled so data are different)
    station_group = m.add_station("mt02")
    fc_group = (
        station_group.fourier_coefficients_group.add_fc_group(
            "processing_run_01"
        )
    )
    decimation_level = fc_group.add_decimation_level("3")
    expected_sr_decimation_level = 0.015380859375
    decimation_level.from_xarray(ds * 1.1, expected_sr_decimation_level)
    decimation_level.update_metadata()
    fc_group.update_metadata()

    m.close_mth5()
    return m


def create_xarray_test_dataset_with_various_dtypes():
    t0 = pd.Timestamp("now")
    t1 = t0 + pd.Timedelta(seconds=1)
    t2 = t1 + pd.Timedelta(seconds=1)

    j = np.complex128(0 + 1j)
    d = {

        "time": {"dims": ("time"), "data": [t0, t1, t2]},
        "bools": {"dims": ("time"), "data": [True, True, False]},
        "ints": {"dims": ("time"), "data": [10, 20, 30]},
        "floats": {"dims": ("time"), "data": [10., 20., 30.]},
        "complexs": {"dims": ("time"), "data": [j * 10., j * 20., j * 30.]},
    }
    xrds = xr.Dataset.from_dict(d)
    freq = np.array([0.667])
    xrds = xrds.expand_dims({"frequency": freq})
    return xrds

def read_fc_csv(csv_name):
    """
    read csv to xarray
    :param csv_name: CSV File with some stored FC values for testing
    :type csv_name: pathlib.Path
    :return: the data from the csv as an xarray
    :rtype: xarray.core.dataset.Dataset

    """
    df = pd.read_csv(
        csv_name,
        index_col=[0, 1],
        parse_dates=["time"],
        skipinitialspace=True,
    )
    for col in df.columns:
        df[col] = np.complex128(df[col])

    return df.to_xarray()


class TestFCFromXarray(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """
        This should only build the file and then close it.
        Don't want to bake in self.station_group, self.dec_level etc. here,
        because if close and reopen the file these are not valid anymore.

        Returns
        -------

        """
        create_mth5_with_some_test_data()

        # Expected properties of the test dataset stored as csv
        self.ds = read_fc_csv(csv_fn)
        self.expected_sr_decimation_level = 0.015380859375
        self.expected_start = MTime(self.ds.time[0].values)
        self.expected_end = MTime(self.ds.time[-1].values)
        self.expected_window_step = 6144
        self.expected_shape = (6, 64)
        self.expected_time = np.array(
            [
                "1980-01-01T00:00:00.000000000",
                "1980-01-01T01:42:24.000000000",
                "1980-01-01T03:24:48.000000000",
                "1980-01-01T05:07:12.000000000",
                "1980-01-01T06:49:36.000000000",
                "1980-01-01T08:32:00.000000000",
            ],
            dtype="datetime64[ns]",
        )

        self.expected_frequency = np.array(
            [
                0.0,
                0.00012207,
                0.00024414,
                0.00036621,
                0.00048828,
                0.00061035,
                0.00073242,
                0.00085449,
                0.00097656,
                0.00109863,
                0.0012207,
                0.00134277,
                0.00146484,
                0.00158691,
                0.00170898,
                0.00183105,
                0.00195312,
                0.0020752,
                0.00219727,
                0.00231934,
                0.00244141,
                0.00256348,
                0.00268555,
                0.00280762,
                0.00292969,
                0.00305176,
                0.00317383,
                0.0032959,
                0.00341797,
                0.00354004,
                0.00366211,
                0.00378418,
                0.00390625,
                0.00402832,
                0.00415039,
                0.00427246,
                0.00439453,
                0.0045166,
                0.00463867,
                0.00476074,
                0.00488281,
                0.00500488,
                0.00512695,
                0.00524902,
                0.00537109,
                0.00549316,
                0.00561523,
                0.0057373,
                0.00585938,
                0.00598145,
                0.00610352,
                0.00622559,
                0.00634766,
                0.00646973,
                0.0065918,
                0.00671387,
                0.00683594,
                0.00695801,
                0.00708008,
                0.00720215,
                0.00732422,
                0.00744629,
                0.00756836,
                0.00769043,
            ]
        )

    def setUp(self) -> None:
        self.h5_filename = h5_filename# fn_path.joinpath("fc_test.h5")
        self.m = MTH5()
        self.m.file_version = "0.1.0"
        self.m.open_mth5(self.h5_filename)
        self.station_group = self.m.get_station("mt01")
        self.fc_group = (
            self.station_group.fourier_coefficients_group.add_fc_group(
                "processing_run_01"
            )
        )
        self.decimation_level = self.fc_group.get_decimation_level("3")

    def tearDown(self) -> None:
        self.m.close_mth5()

    def test_channel_exists(self):
        self.assertListEqual(
            list(self.ds.data_vars.keys()),
            self.decimation_level.groups_list,
        )

    def test_channel_metadata(self):
        for ch in self.decimation_level.groups_list:
            fc_ch = self.decimation_level.get_channel(ch)
            with self.subTest(f"{ch} name"):
                self.assertEqual(fc_ch.metadata.component, ch)
            with self.subTest(f"{ch} start"):
                self.assertEqual(
                    fc_ch.metadata.time_period.start, self.expected_start
                )
            with self.subTest(f"{ch} end"):
                self.assertEqual(
                    fc_ch.metadata.time_period.end, self.expected_end
                )
            with self.subTest(f"{ch} window_step"):
                self.assertEqual(
                    fc_ch.metadata.sample_rate_window_step,
                    self.expected_window_step,
                )
            with self.subTest(f"{ch} sr_decimation_level"):
                self.assertEqual(
                    fc_ch.metadata.sample_rate_decimation_level,
                    self.expected_sr_decimation_level,
                )
            with self.subTest(f"{ch} shape"):
                self.assertTupleEqual(
                    fc_ch.hdf5_dataset.shape, self.expected_shape
                )

            with self.subTest(f"{ch} time"):
                self.assertTrue((fc_ch.time == self.expected_time).all())
            with self.subTest(f"{ch} frequency"):
                self.assertTrue(
                    np.isclose(fc_ch.frequency, self.expected_frequency).all()
                )
            with self.subTest("metadata and table sample rates agree"):
                df = self.decimation_level.channel_summary
                assert (df.sample_rate_decimation_level.iloc[0] == self.expected_sr_decimation_level)

    def test_to_xarray(self):
        da = self.decimation_level.to_xarray()

        self.assertEqual(da, self.ds)

    def test_ch_to_xarray(self):
        fc_ch = self.decimation_level.get_channel("ex")
        ch_da = fc_ch.to_xarray()

        with self.subTest("time"):
            self.assertTrue((ch_da.time.values == self.expected_time).all())
        with self.subTest("frequency"):
            self.assertTrue(
                np.isclose(ch_da.frequency, self.expected_frequency).all()
            )
        with self.subTest("name"):
            self.assertTrue("ex", ch_da.name)

        with self.subTest("ex start"):
            self.assertEqual(
                ch_da.attrs["time_period.start"], self.expected_start
            )
        with self.subTest("ex end"):
            self.assertEqual(ch_da.attrs["time_period.end"], self.expected_end)
        with self.subTest("ex window_step"):
            self.assertEqual(
                ch_da.attrs["sample_rate_window_step"],
                self.expected_window_step,
            )
        with self.subTest("ex sr_decimation_level"):
            self.assertEqual(
                ch_da.attrs["sample_rate_decimation_level"],
                self.expected_sr_decimation_level,
            )
        with self.subTest("ex shape"):
            self.assertTupleEqual(ch_da.shape, self.expected_shape)

    def test_can_update_decimation_level_metadata(self):
        window_type = "hann"
        # set the window typw
        self.decimation_level.metadata.window.type = window_type
        # assert that the updated value is true
        with self.subTest("window.type is set"):
            self.assertEqual(
                self.decimation_level.metadata.window.type, window_type
            )
        self.decimation_level.update_metadata()
        self.decimation_level.write_metadata()

        self.fc_group.update_metadata()
        self.fc_group.write_metadata()

        tmp = self.fc_group.get_decimation_level("3")
        with self.subTest("get_decimation_level.metadata.window.type"):
            self.assertEqual(tmp.metadata.window.type, window_type)

    def test_from_xarray_dtypes(self):
        """
        Intialize a dummy h5 and create an FC level
        - in that fc level we will create a container (say "features")
        - within that level we will store channels
            - bool_channel
            - int
            - float_ch
            - complex_channel
        - Then update metadata/close file
        - Then open file and check that dtypes are expected
        Returns
        -------

        """
        dec_level_name = "ringo"
        fc_metadata = self.decimation_level.metadata.copy()
        fc_metadata.id = dec_level_name
        fc_decimation_level = self.fc_group.add_decimation_level(dec_level_name, decimation_level_metadata=fc_metadata)
        xrds = create_xarray_test_dataset_with_various_dtypes()

        fc_decimation_level.from_xarray(xrds, fc_metadata.sample_rate)
        fc_decimation_level.update_metadata()
        self.fc_group.update_metadata()
        self.m.close_mth5()
        self.setUp()

        reopened_dec_level = self.fc_group.get_decimation_level(dec_level_name)
        xrds2 = reopened_dec_level.to_xarray()
        assert xrds2.bools.dtype == bool
        assert xrds2.ints.dtype == int
        assert xrds2.floats.dtype == np.float64
        assert xrds2.complexs.dtype == np.complex128

    def test_multi_station_spectrogram(self):
        """
        TODO: This could be moved to fc_tools.
        It is here because there was a handy dataset already available here.

        """
        fc_run_chunks = []
        for station_id in self.m.station_list:  # ["mt01", "mt02"]
            fcrc = FCRunChunk(
                station_id=station_id,
                run_id="processing_run_01",
                decimation_level_id="3",
                # start="2023-10-05T20:03:00",
                start="",
                end="",
                channels=[],
            )
            fc_run_chunks.append(fcrc)

        # TODO: These tests should go in their own module, but need test dataset (issue #227)
        xrds = make_multistation_spectrogram(self.m, fc_run_chunks, rtype="xrds")
        assert isinstance(xrds, xarray.Dataset)
        mvds = make_multistation_spectrogram(self.m, fc_run_chunks, rtype=None)
        assert isinstance(mvds, MultivariateDataset)

        # Test that channels method is unpacking the right number of valuesl
        assert len(mvds.channels) == len(xrds.data_vars)
        # TODO : add tests for Multivariate Data

        assert len(mvds.stations) != 0

        total_channels = 0
        for station_id in mvds.stations:
            total_channels += len(mvds.station_channels(station_id))
        assert total_channels == len(mvds.channels)

        # print("OK")


    @classmethod
    def tearDownClass(self):
        # self.m.close_mth5()
        h5_filename.unlink()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
