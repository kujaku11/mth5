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

    def test_validate_channel_type_auxiliary(self):
        self.assertEqual("Auxiliary", self.ts._validate_channel_type("frogs"))

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
        self.assertDictEqual(
            self.ts.run_metadata.to_dict(single=True),
            self.ts._validate_run_metadata(self.ts.run_metadata).to_dict(
                single=True
            ),
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

        self.assertRaises(TypeError, set_ts, 10)

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
                self.assertRaises(ValueError, set_comp, ch)

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
                "2020-01-01T12:00:00", end="2020-01-01T12:00:02.937500"
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


class TestChannelTS2ObspyTrace(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.ch = timeseries.ChannelTS(
            "auxiliary",
            data=np.random.rand(4096),
            channel_metadata={
                "auxiliary": {
                    "time_period.start": "2020-01-01T12:00:00",
                    "sample_rate": 8,
                    "component": "temp",
                    "type": "temperature",
                }
            },
            station_metadata={"Station": {"id": "mt01"}},
            run_metadata={"Run": {"id": "0001"}},
        )

    def test_to_obspy_trace(self):
        tr = self.ch.to_obspy_trace()
        with self.subTest("network"):
            self.assertEqual("", tr.stats.network)
        with self.subTest("station"):
            self.assertEqual(self.ch.station_metadata.id, tr.stats.station)
        with self.subTest("location"):
            self.assertEqual("", tr.stats.location)
        with self.subTest("channel"):
            self.assertEqual("MKN", tr.stats.channel)
        with self.subTest("start"):
            self.assertEqual(self.ch.start.isoformat(), tr.stats.starttime)
        with self.subTest("end"):
            self.assertEqual(self.ch.end.isoformat(), tr.stats.endtime)
        with self.subTest("sample_rate"):
            self.assertEqual(self.ch.sample_rate, tr.stats.sampling_rate)
        with self.subTest("delta"):
            self.assertEqual(1 / self.ch.sample_rate, tr.stats.delta)
        with self.subTest("npts"):
            self.assertEqual(self.ch.ts.size, tr.stats.npts)
        with self.subTest("calib"):
            self.assertEqual(1.0, tr.stats.calib)
        with self.subTest("Data"):
            self.assertTrue(np.allclose(self.ch.ts, tr.data))

    def test_from_obspy_trace(self):
        tr = self.ch.to_obspy_trace()
        new_ch = timeseries.ChannelTS()
        new_ch.from_obspy_trace(tr)

        with self.subTest("station"):
            self.assertEqual(
                self.ch.station_metadata.id, new_ch.station_metadata.id
            )
        with self.subTest("channel"):
            self.assertEqual("temperaturex", new_ch.component)
        with self.subTest("channel metadata type"):
            self.assertEqual("temperature", new_ch.channel_metadata.type)
        with self.subTest("start"):
            self.assertEqual(self.ch.start, new_ch.start)
        with self.subTest("end"):
            self.assertEqual(self.ch.end, new_ch.end)
        with self.subTest("sample_rate"):
            self.assertEqual(self.ch.sample_rate, new_ch.sample_rate)
        with self.subTest("npts"):
            self.assertEqual(self.ch.ts.size, new_ch.ts.size)
        with self.subTest("Data"):
            self.assertTrue(np.allclose(self.ch.ts, new_ch.ts))


class TestAddChannels(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.survey_metadata = metadata.Survey(id="test")

        self.station_metadata = metadata.Station(id="mt01")
        self.station_metadata.location.latitude = 40
        self.station_metadata.location.longitude = -112
        self.station_metadata.location.elevation = 120

        self.run_metadata = metadata.Run(id="001")

        self.channel_metadata = metadata.Electric(
            component="ex", sample_rate=1
        )
        self.channel_metadata.time_period.start = "2020-01-01T00:00:00+00:00"
        self.channel_metadata.time_period.end = "2020-01-01T00:00:59+00:00"

        self.channel_metadata2 = metadata.Electric(
            component="ex", sample_rate=1
        )
        self.channel_metadata2.time_period.start = "2020-01-01T00:01:10"
        self.channel_metadata2.time_period.end = "2020-01-01T00:02:09"

        self.combined_start = "2020-01-01T00:00:00+00:00"
        self.combined_end = "2020-01-01T00:02:09+00:00"

        self.ex1 = timeseries.ChannelTS(
            channel_type="electric",
            data=np.linspace(0, 59, 60),
            channel_metadata=self.channel_metadata,
            survey_metadata=self.survey_metadata,
            station_metadata=self.station_metadata,
            run_metadata=self.run_metadata,
        )
        self.ex2 = timeseries.ChannelTS(
            channel_type="electric",
            data=np.linspace(70, 69 + 60, 60),
            channel_metadata=self.channel_metadata2,
        )

        self.combined_ex = self.ex1 + self.ex2

    def test_copy(self):
        ex1_copy = self.ex1.copy()

        self.assertEqual(self.ex1, ex1_copy)

    def test_survey_metadata(self):
        with self.subTest("id"):
            self.assertEqual(
                self.survey_metadata.id, self.combined_ex.survey_metadata.id
            )
        with self.subTest("start"):
            self.assertEqual(
                self.combined_start,
                self.combined_ex.survey_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.assertEqual(
                self.combined_end,
                self.combined_ex.survey_metadata.time_period.end,
            )

    def test_station_metadata(self):
        for key in [
            "id",
            "location.latitude",
            "location.longitude",
            "location.elevation",
        ]:
            with self.subTest(key):
                self.assertEqual(
                    self.station_metadata.get_attr_from_name(key),
                    self.combined_ex.station_metadata.get_attr_from_name(key),
                )

        with self.subTest("start"):
            self.assertEqual(
                self.combined_start,
                self.combined_ex.station_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.assertEqual(
                self.combined_end,
                self.combined_ex.station_metadata.time_period.end,
            )

    def test_run_metadata(self):
        with self.subTest("id"):
            self.assertEqual(
                self.run_metadata.id, self.combined_ex.run_metadata.id
            )
        with self.subTest("start"):
            self.assertEqual(
                self.combined_start,
                self.combined_ex.run_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.assertEqual(
                self.combined_end,
                self.combined_ex.run_metadata.time_period.end,
            )

    def test_channel_metadata(self):
        with self.subTest("component"):
            self.assertEqual(
                self.channel_metadata.component,
                self.combined_ex.channel_metadata.component,
            )
        with self.subTest("start"):
            self.assertEqual(
                self.combined_start,
                self.combined_ex.channel_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.assertEqual(
                self.combined_end,
                self.combined_ex.channel_metadata.time_period.end,
            )

    def test_data(self):
        data = np.arange(130)
        self.assertTrue(np.all(data == self.combined_ex.ts))

    def test_data_size(self):
        self.assertEqual(130, self.combined_ex.ts.size)

    def test_has_data(self):
        self.assertEqual(True, self.combined_ex.has_data())


class TestMergeChannels(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.survey_metadata = metadata.Survey(id="test")

        self.station_metadata = metadata.Station(id="mt01")
        self.station_metadata.location.latitude = 40
        self.station_metadata.location.longitude = -112
        self.station_metadata.location.elevation = 120

        self.run_metadata = metadata.Run(id="001")

        self.channel_metadata = metadata.Electric(
            component="ex", sample_rate=10
        )
        self.channel_metadata.time_period.start = "2020-01-01T00:00:00+00:00"
        self.channel_metadata.time_period.end = "2020-01-01T00:00:59+00:00"

        self.channel_metadata2 = metadata.Electric(
            component="ex", sample_rate=10
        )
        self.channel_metadata2.time_period.start = "2020-01-01T00:01:10"
        self.channel_metadata2.time_period.end = "2020-01-01T00:02:09"

        self.combined_start = "2020-01-01T00:00:00+00:00"
        self.combined_end = "2020-01-01T00:02:09+00:00"

        self.ex1 = timeseries.ChannelTS(
            channel_type="electric",
            data=np.linspace(0, 59, 600),
            channel_metadata=self.channel_metadata,
            survey_metadata=self.survey_metadata,
            station_metadata=self.station_metadata,
            run_metadata=self.run_metadata,
        )
        self.ex2 = timeseries.ChannelTS(
            channel_type="electric",
            data=np.linspace(70, 69 + 60, 600),
            channel_metadata=self.channel_metadata2,
        )

        self.combined_ex = self.ex1.merge(self.ex2, new_sample_rate=1)

    def test_survey_metadata(self):
        with self.subTest("id"):
            self.assertEqual(
                self.survey_metadata.id, self.combined_ex.survey_metadata.id
            )
        with self.subTest("start"):
            self.assertEqual(
                self.combined_start,
                self.combined_ex.survey_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.assertEqual(
                self.combined_end,
                self.combined_ex.survey_metadata.time_period.end,
            )

    def test_station_metadata(self):
        for key in [
            "id",
            "location.latitude",
            "location.longitude",
            "location.elevation",
        ]:
            with self.subTest(key):
                self.assertEqual(
                    self.station_metadata.get_attr_from_name(key),
                    self.combined_ex.station_metadata.get_attr_from_name(key),
                )

        with self.subTest("start"):
            self.assertEqual(
                self.combined_start,
                self.combined_ex.station_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.assertEqual(
                self.combined_end,
                self.combined_ex.station_metadata.time_period.end,
            )

    def test_run_metadata(self):
        with self.subTest("id"):
            self.assertEqual(
                self.run_metadata.id, self.combined_ex.run_metadata.id
            )
        with self.subTest("start"):
            self.assertEqual(
                self.combined_start,
                self.combined_ex.run_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.assertEqual(
                self.combined_end,
                self.combined_ex.run_metadata.time_period.end,
            )

    def test_channel_metadata(self):
        with self.subTest("component"):
            self.assertEqual(
                self.channel_metadata.component,
                self.combined_ex.channel_metadata.component,
            )
        with self.subTest("start"):
            self.assertEqual(
                self.combined_start,
                self.combined_ex.channel_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.assertEqual(
                self.combined_end,
                self.combined_ex.channel_metadata.time_period.end,
            )

    def test_data(self):
        data = np.array(
            [
                [
                    0.0,
                    0.98497496,
                    1.96994992,
                    2.95492487,
                    3.93989983,
                    4.92487479,
                    5.90984975,
                    6.89482471,
                    7.87979967,
                    8.86477462,
                    9.84974958,
                    10.83472454,
                    11.8196995,
                    12.80467446,
                    13.78964942,
                    14.77462437,
                    15.75959933,
                    16.74457429,
                    17.72954925,
                    18.71452421,
                    19.69949917,
                    20.68447412,
                    21.66944908,
                    22.65442404,
                    23.639399,
                    24.62437396,
                    25.60934891,
                    26.59432387,
                    27.57929883,
                    28.56427379,
                    29.54924875,
                    30.53422371,
                    31.51919866,
                    32.50417362,
                    33.48914858,
                    34.47412354,
                    35.4590985,
                    36.44407346,
                    37.42904841,
                    38.41402337,
                    39.39899833,
                    40.38397329,
                    41.36894825,
                    42.35392321,
                    43.33889816,
                    44.32387312,
                    45.30884808,
                    46.29382304,
                    47.278798,
                    48.26377295,
                    49.24874791,
                    50.23372287,
                    51.21869783,
                    52.20367279,
                    53.18864775,
                    54.1736227,
                    55.15859766,
                    56.14357262,
                    57.12854758,
                    58.11352254,
                    59.1194679,
                    60.12541326,
                    61.13135862,
                    62.13730398,
                    63.14324934,
                    64.1491947,
                    65.15514006,
                    66.16108543,
                    67.16703079,
                    68.17297615,
                    69.17892151,
                    70.16844791,
                    71.14234838,
                    72.11462901,
                    73.09783692,
                    74.0635483,
                    75.04416454,
                    76.01248987,
                    76.98908698,
                    77.96251863,
                    78.93444956,
                    79.91091112,
                    80.88183766,
                    81.85762617,
                    82.83023514,
                    83.80406446,
                    84.77833522,
                    85.75111637,
                    86.72575336,
                    87.69873647,
                    88.67281592,
                    89.6464776,
                    90.61995581,
                    91.59401218,
                    92.56735351,
                    93.54130646,
                    94.51492439,
                    95.48852248,
                    96.46247091,
                    97.43585606,
                    98.40983087,
                    99.3834074,
                    100.35699165,
                    101.33108209,
                    102.30416022,
                    103.27857937,
                    104.2516794,
                    105.22560032,
                    106.19969464,
                    107.17226152,
                    108.14775621,
                    109.1193389,
                    110.09488059,
                    111.06777989,
                    112.04050508,
                    113.01738044,
                    113.98587942,
                    114.96585166,
                    115.93390731,
                    116.91049487,
                    117.88594881,
                    118.85205662,
                    119.83952487,
                    120.793457,
                    121.7899573,
                    122.74966675,
                    123.72735148,
                    124.73152643,
                    125.63425786,
                    126.74465939,
                ]
            ]
        )
        self.assertTrue(np.allclose(data, self.combined_ex.ts))

    def test_data_size(self):
        self.assertEqual(130, self.combined_ex.ts.size)

    def test_has_data(self):
        self.assertEqual(True, self.combined_ex.has_data())


# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
