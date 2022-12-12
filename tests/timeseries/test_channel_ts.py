# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:32:55 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# imports
# =============================================================================

import unittest

import numpy as np
import pandas as pd

from mth5 import timeseries
from mth5.utils.exceptions import MTTSError

from mt_metadata import timeseries as metadata
from mt_metadata.timeseries.filters import CoefficientFilter

# =============================================================================
#
# =============================================================================


class TestChannelTS(unittest.TestCase):
    def setUp(self):
        self.ts = timeseries.ChannelTS("auxiliary")
        self.maxDiff = None

    def test_input_type_electric(self):
        self.ts = timeseries.ChannelTS("electric")

        electric_meta = metadata.Electric(type="electric")
        self.assertDictEqual(
            self.ts.channel_metadata.to_dict(), electric_meta.to_dict()
        )

    def test_input_type_magnetic(self):
        self.ts = timeseries.ChannelTS("magnetic")

        magnetic_meta = metadata.Magnetic(type="magnetic")
        self.assertDictEqual(
            self.ts.channel_metadata.to_dict(), magnetic_meta.to_dict()
        )

    def test_input_type_auxiliary(self):
        self.ts = timeseries.ChannelTS("auxiliary")

        auxiliary_meta = metadata.Auxiliary(type="auxiliary")
        self.assertDictEqual(
            self.ts.channel_metadata.to_dict(), auxiliary_meta.to_dict()
        )

    def test_input_type_fail(self):
        self.assertRaises(ValueError, timeseries.ChannelTS, "temperature")

    def test_set_channel_fail(self):
        self.assertRaises(
            TypeError, timeseries.ChannelTS, **{"channel_metadata": []}
        )

    def test_set_run_metadata_fail(self):
        self.assertRaises(
            TypeError, timeseries.ChannelTS, **{"run_metadata": []}
        )

    def test_set_station_metadata_fail(self):
        self.assertRaises(
            TypeError, timeseries.ChannelTS, **{"station_metadata": []}
        )

    def test_validate_channel_type(self):
        for ch in ["electric", "magnetic", "auxiliary"]:
            with self.subTest(ch):
                self.assertEqual(
                    ch.capitalize(), self.ts._validate_channel_type(ch)
                )

    def test_validate_channel_type_fail(self):
        def set_ch_type(value):
            self.ts._validate_channel_type(value)

        self.assertRaises(ValueError, set_ch_type, "frogs")

    def test_validate_channel_metadata(self):
        self.assertEqual(
            self.ts.channel_metadata,
            self.ts._validate_channel_metadata(self.ts.channel_metadata),
        )

    def test_validate_channel_metadata_from_dict(self):
        self.assertEqual(
            self.ts.channel_metadata,
            self.ts._validate_channel_metadata({"type": "auxiliary"}),
        )

    def test_validate_run_metadata(self):
        self.assertEqual(
            self.ts.run_metadata,
            self.ts._validate_run_metadata(self.ts.run_metadata),
        )

    def test_validate_run_metadata_from_dict(self):
        self.assertEqual(
            metadata.Run(id="0"),
            self.ts._validate_run_metadata({"id": "0"}),
        )

    def test_validate_station_metadata(self):
        self.assertEqual(
            self.ts.station_metadata,
            self.ts._validate_station_metadata(self.ts.station_metadata),
        )

    def test_validate_station_metadata_from_dict(self):
        self.assertEqual(
            metadata.Station(id="0"),
            self.ts._validate_station_metadata({"id": "0"}),
        )

    def test_validate_survey_metadata(self):
        self.assertEqual(
            self.ts.survey_metadata,
            self.ts._validate_survey_metadata(self.ts.survey_metadata),
        )

    def test_validate_survey_metadata_from_dict(self):
        self.assertEqual(
            metadata.Survey(id="0"),
            self.ts._validate_survey_metadata({"id": "0"}),
        )

    def test_str(self):
        lines = [
            f"Survey:       {self.ts.survey_metadata.id}",
            f"Station:      {self.ts.station_metadata.id}",
            f"Run:          {self.ts.run_metadata.id}",
            f"Channel Type: {self.ts.channel_type}",
            f"Component:    {self.ts.component}",
            f"Sample Rate:  {self.ts.sample_rate}",
            f"Start:        {self.ts.start}",
            f"End:          {self.ts.end}",
            f"N Samples:    {self.ts.n_samples}",
        ]

        test_str = "\n\t".join(["Channel Summary:"] + lines)
        self.assertEqual(test_str, self.ts.__str__())

    def test_repr(self):
        lines = [
            f"Survey:       {self.ts.survey_metadata.id}",
            f"Station:      {self.ts.station_metadata.id}",
            f"Run:          {self.ts.run_metadata.id}",
            f"Channel Type: {self.ts.channel_type}",
            f"Component:    {self.ts.component}",
            f"Sample Rate:  {self.ts.sample_rate}",
            f"Start:        {self.ts.start}",
            f"End:          {self.ts.end}",
            f"N Samples:    {self.ts.n_samples}",
        ]

        test_str = "\n\t".join(["Channel Summary:"] + lines)
        self.assertEqual(test_str, self.ts.__repr__())

    def test_intialize_with_metadata(self):
        self.ts = timeseries.ChannelTS(
            "electric", channel_metadata={"electric": {"component": "ex"}}
        )
        with self.subTest(name="component in metadata"):
            self.assertEqual(self.ts.channel_metadata.component, "ex")
        with self.subTest(name="compnent in attrs"):
            self.assertEqual(self.ts._ts.attrs["component"], "ex")

    def test_equal(self):
        self.assertTrue(self.ts == self.ts)

    def test_not_equal(self):
        x = timeseries.ChannelTS(channel_type="electric")
        self.assertFalse(self.ts == x)

    def test_less_than(self):
        x = timeseries.ChannelTS(channel_type="electric")
        x.start = "2020-01-01T12:00:00"
        self.assertFalse(self.ts < x)

    def test_greater_than(self):
        x = timeseries.ChannelTS(channel_type="electric")
        x.start = "2020-01-01T12:00:00"
        self.assertTrue(self.ts > x)

    def test_numpy_input(self):
        self.ts.channel_metadata.sample_rate = 1.0
        self.ts._update_xarray_metadata()

        self.ts.ts = np.random.rand(4096)
        end = self.ts.channel_metadata.time_period._start_dt + (4096 - 1)

        # check to make sure the times align
        with self.subTest(name="is aligned"):
            self.assertEqual(
                self.ts._ts.coords.to_index()[0].isoformat(),
                self.ts.channel_metadata.time_period._start_dt.iso_no_tz,
            )
        with self.subTest(name="has index"):
            self.assertEqual(
                self.ts._ts.coords.to_index()[-1].isoformat(), end.iso_no_tz
            )
        with self.subTest(name="has n samples"):
            self.assertEqual(self.ts.n_samples, 4096)

    def test_numpy_input_fail(self):
        self.ts.channel_metadata.sample_rate = 1.0

        def set_ts(ts_obj, ts_arr):
            ts_obj.ts = ts_arr

        self.assertRaises(ValueError, set_ts, self.ts, np.random.rand(2, 4096))

    def test_list_input(self):
        self.ts.channel_metadata.sample_rate = 1.0

        self.ts.ts = np.random.rand(4096).tolist()
        end = self.ts.channel_metadata.time_period._start_dt + (4096 - 1)

        # check to make sure the times align
        with self.subTest(name="is aligned"):
            self.assertEqual(
                self.ts._ts.coords.to_index()[0].isoformat(),
                self.ts.channel_metadata.time_period._start_dt.iso_no_tz,
            )
        with self.subTest(name="has index"):
            self.assertEqual(
                self.ts._ts.coords.to_index()[-1].isoformat(), end.iso_no_tz
            )
        with self.subTest(name="has n samples"):
            self.assertEqual(self.ts.n_samples, 4096)

    def test_input_fail(self):
        def set_ts(value):
            self.ts.ts = value

        self.assertRaises(MTTSError, set_ts, 10)

    def test_df_without_index_input(self):
        self.ts.channel_metadata.sample_rate = 1.0

        self.ts._update_xarray_metadata()

        self.ts.ts = pd.DataFrame({"data": np.random.rand(4096)})
        end = self.ts.channel_metadata.time_period._start_dt + (4096 - 1)

        # check to make sure the times align
        with self.subTest(name="is aligned"):
            self.assertEqual(
                self.ts._ts.coords.to_index()[0].isoformat(),
                self.ts.channel_metadata.time_period._start_dt.iso_no_tz,
            )
        with self.subTest(name="has index"):
            self.assertEqual(
                self.ts._ts.coords.to_index()[-1].isoformat(), end.iso_no_tz
            )
        with self.subTest(name="has n samples"):
            self.assertEqual(self.ts.n_samples, 4096)

    def test_df_with_index_input(self):
        n_samples = 4096
        self.ts.ts = pd.DataFrame(
            {"data": np.random.rand(n_samples)},
            index=pd.date_range(
                start="2020-01-02T12:00:00",
                periods=n_samples,
                end="2020-01-02T12:00:01",
            ),
        )

        # check to make sure the times align
        with self.subTest(name="is aligned"):
            self.assertEqual(
                self.ts._ts.coords.to_index()[0].isoformat(),
                self.ts.channel_metadata.time_period._start_dt.iso_no_tz,
            )
        # check to make sure the times align
        with self.subTest(name="same end"):
            self.assertEqual(
                self.ts._ts.coords.to_index()[-1].isoformat(),
                self.ts.channel_metadata.time_period._end_dt.iso_no_tz,
            )
        with self.subTest(name="sample rate"):
            self.assertEqual(self.ts.sample_rate, 4096.0)
        with self.subTest(name="has n samples"):
            self.assertEqual(self.ts.n_samples, n_samples)

    def test_set_component(self):
        self.ts = timeseries.ChannelTS(
            "electric", channel_metadata={"electric": {"component": "ex"}}
        )

        def set_comp(comp):
            self.ts.component = comp

        for ch in ["hx", "bx", "temperature"]:
            with self.subTest(name=f"fail {ch}"):
                self.assertRaises(MTTSError, set_comp, ch)

    def test_change_sample_rate(self):
        self.ts.sample_rate = 16
        self.ts.start = "2020-01-01T12:00:00"
        self.ts.ts = np.arange(4096)

        self.assertEqual(self.ts.sample_rate, 16.0)

        with self.subTest(name="sample_interval"):
            self.assertEqual(self.ts.sample_interval, 1.0 / 16.0)
        self.ts.sample_rate = 8
        with self.subTest("8"):
            self.assertEqual(self.ts.sample_rate, 8.0)
        with self.subTest("n_samples"):
            self.assertEqual(self.ts.n_samples, 4096)

        with self.subTest(name="sample_interval"):
            self.assertEqual(self.ts.sample_interval, 1.0 / 8.0)

    def test_to_xarray(self):
        self.ts.sample_rate = 16
        self.ts.start = "2020-01-01T12:00:00"
        self.ts.ts = np.arange(4096)
        self.ts.station_metadata.id = "mt01"
        self.ts.run_metadata.id = "mt01a"

        ts_xr = self.ts.to_xarray()

        with self.subTest("station ID"):
            self.assertEqual(ts_xr.attrs["station.id"], "mt01")
        with self.subTest("run ID"):
            self.assertEqual(ts_xr.attrs["run.id"], "mt01a")
        with self.subTest("sample rate"):
            self.assertEqual(ts_xr.sample_rate, 16)
        with self.subTest("start"):
            self.assertEqual(
                ts_xr.coords["time"].to_index()[0].isoformat(),
                ts_xr.attrs["time_period.start"].split("+")[0],
            )
        with self.subTest("end"):
            self.assertEqual(
                ts_xr.coords["time"].to_index()[-1].isoformat(),
                ts_xr.attrs["time_period.end"].split("+")[0],
            )

    def test_xarray_input(self):
        ch = timeseries.ChannelTS(
            "auxiliary",
            data=np.random.rand(4096),
            channel_metadata={
                "auxiliary": {
                    "time_period.start": "2020-01-01T12:00:00",
                    "sample_rate": 8,
                    "component": "temp",
                }
            },
            station_metadata={"Station": {"id": "mt01"}},
            run_metadata={"Run": {"id": "0001"}},
        )
        with self.subTest("station ID"):
            self.assertEqual(ch.channel_type.lower(), "auxiliary")
        self.ts.ts = ch.to_xarray()

        with self.subTest("run ID"):
            self.assertEqual(self.ts.run_metadata.id, "0001")
        with self.subTest("station ID"):
            self.assertEqual(self.ts.station_metadata.id, "mt01")
        with self.subTest("start"):
            self.assertEqual(self.ts.start, "2020-01-01T12:00:00+00:00")

    def test_time_slice(self):
        self.ts.component = "temp"
        self.ts.sample_rate = 16
        self.ts.start = "2020-01-01T12:00:00"
        self.ts.ts = np.arange(4096)

        with self.subTest(name="nsamples"):
            new_ts = self.ts.get_slice("2020-01-01T12:00:00", n_samples=48)
            self.assertEqual(new_ts.ts.size, 48)
        with self.subTest(name="end time"):
            new_ts = self.ts.get_slice(
                "2020-01-01T12:00:00", end="2020-01-01T12:00:03"
            )
            self.assertEqual(new_ts.ts.size, 48)

    def test_time_slice_metadata(self):
        self.ts.component = "temp"
        self.ts.sample_rate = 16
        self.ts.start = "2020-01-01T12:00:00"
        self.ts.ts = np.arange(4096)
        self.ts.channel_metadata.filter.name = "example_filter"
        self.ts.channel_response_filter.filters_list.append(
            CoefficientFilter(name="example_filter", gain=10)
        )
        new_ts = self.ts.get_slice("2020-01-01T12:00:00", n_samples=48)

        with self.subTest("metadata"):
            self.assertEqual(new_ts.channel_metadata, new_ts.channel_metadata)

        with self.subTest("channel_response"):
            self.assertEqual(
                new_ts.channel_response_filter, self.ts.channel_response_filter
            )

        with self.subTest("run metadata"):
            self.assertEqual(new_ts.run_metadata, new_ts.run_metadata)

        with self.subTest("station metadata"):
            self.assertEqual(new_ts.station_metadata, new_ts.station_metadata)


# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
