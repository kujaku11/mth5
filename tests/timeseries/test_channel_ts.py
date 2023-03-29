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
                    -1.97951176e-02,
                    9.69731280e-01,
                    1.94363175e00,
                    2.91591238e00,
                    3.89912030e00,
                    4.86483168e00,
                    5.84544792e00,
                    6.81377324e00,
                    7.79037036e00,
                    8.76380201e00,
                    9.73573293e00,
                    1.07121945e01,
                    1.16831210e01,
                    1.26589095e01,
                    1.36315185e01,
                    1.46053478e01,
                    1.55796186e01,
                    1.65523997e01,
                    1.75270367e01,
                    1.85000198e01,
                    1.94740993e01,
                    2.04477610e01,
                    2.14212392e01,
                    2.23952956e01,
                    2.33686369e01,
                    2.43425898e01,
                    2.53162078e01,
                    2.62898058e01,
                    2.72637543e01,
                    2.82371394e01,
                    2.92111142e01,
                    3.01846908e01,
                    3.11582750e01,
                    3.21323655e01,
                    3.31054436e01,
                    3.40798627e01,
                    3.50529628e01,
                    3.60268837e01,
                    3.70009780e01,
                    3.79735449e01,
                    3.89490396e01,
                    3.99206223e01,
                    4.08961640e01,
                    4.18690633e01,
                    4.28417885e01,
                    4.38186638e01,
                    4.47871628e01,
                    4.57671350e01,
                    4.67351907e01,
                    4.77117782e01,
                    4.86872322e01,
                    4.96533400e01,
                    5.06408082e01,
                    5.15947404e01,
                    5.25912407e01,
                    5.35509501e01,
                    5.45286348e01,
                    5.55328098e01,
                    5.64355412e01,
                    5.75459428e01,
                    5.86034863e01,
                    5.96610298e01,
                    6.07185733e01,
                    6.17761169e01,
                    6.28336604e01,
                    6.38912039e01,
                    6.49487474e01,
                    6.60062909e01,
                    6.70638345e01,
                    6.81213780e01,
                    6.91789215e01,
                    7.01684479e01,
                    7.11423484e01,
                    7.21146290e01,
                    7.30978369e01,
                    7.40635483e01,
                    7.50441645e01,
                    7.60124899e01,
                    7.69890870e01,
                    7.79625186e01,
                    7.89344496e01,
                    7.99109111e01,
                    8.08818377e01,
                    8.18576262e01,
                    8.28302351e01,
                    8.38040645e01,
                    8.47783352e01,
                    8.57511164e01,
                    8.67257534e01,
                    8.76987365e01,
                    8.86728159e01,
                    8.96464776e01,
                    9.06199558e01,
                    9.15940122e01,
                    9.25673535e01,
                    9.35413065e01,
                    9.45149244e01,
                    9.54885225e01,
                    9.64624709e01,
                    9.74358561e01,
                    9.84098309e01,
                    9.93834074e01,
                    1.00356992e02,
                    1.01331082e02,
                    1.02304160e02,
                    1.03278579e02,
                    1.04251679e02,
                    1.05225600e02,
                    1.06199695e02,
                    1.07172262e02,
                    1.08147756e02,
                    1.09119339e02,
                    1.10094881e02,
                    1.11067780e02,
                    1.12040505e02,
                    1.13017380e02,
                    1.13985879e02,
                    1.14965852e02,
                    1.15933907e02,
                    1.16910495e02,
                    1.17885949e02,
                    1.18852057e02,
                    1.19839525e02,
                    1.20793457e02,
                    1.21789957e02,
                    1.22749667e02,
                    1.23727351e02,
                    1.24731526e02,
                    1.25634258e02,
                    1.26744659e02,
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
