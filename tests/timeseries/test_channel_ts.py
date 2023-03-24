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


class TestAddChannels(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.survey_metadata = metadata.Survey(id="test")

        self.station_metadata = metadata.Station(id="mt01")
        self.station_metadata.location.latitude = 40
        self.station_metadata.location.longitude = -112
        self.station_metadata.location.elevation = 120

        self.run_metadata = metadata.Run(id="001")

        self.channel_metadata = metadata.Electric(component="ex", sample_rate=1)
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
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                    19.0,
                    20.0,
                    21.0,
                    22.0,
                    23.0,
                    24.0,
                    25.0,
                    26.0,
                    27.0,
                    28.0,
                    29.0,
                    30.0,
                    31.0,
                    32.0,
                    33.0,
                    34.0,
                    35.0,
                    36.0,
                    37.0,
                    38.0,
                    39.0,
                    40.0,
                    41.0,
                    42.0,
                    43.0,
                    44.0,
                    45.0,
                    46.0,
                    47.0,
                    48.0,
                    49.0,
                    50.0,
                    51.0,
                    52.0,
                    53.0,
                    54.0,
                    55.0,
                    56.0,
                    57.0,
                    58.0,
                    59.0,
                    59.0,
                    59.0,
                    59.0,
                    59.0,
                    59.0,
                    70.0,
                    70.0,
                    70.0,
                    70.0,
                    70.0,
                    70.0,
                    71.0,
                    72.0,
                    73.0,
                    74.0,
                    75.0,
                    76.0,
                    77.0,
                    78.0,
                    79.0,
                    80.0,
                    81.0,
                    82.0,
                    83.0,
                    84.0,
                    85.0,
                    86.0,
                    87.0,
                    88.0,
                    89.0,
                    90.0,
                    91.0,
                    92.0,
                    93.0,
                    94.0,
                    95.0,
                    96.0,
                    97.0,
                    98.0,
                    99.0,
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                    110.0,
                    111.0,
                    112.0,
                    113.0,
                    114.0,
                    115.0,
                    116.0,
                    117.0,
                    118.0,
                    119.0,
                    120.0,
                    121.0,
                    122.0,
                    123.0,
                    124.0,
                    125.0,
                    126.0,
                    127.0,
                    128.0,
                    129.0,
                ]
            ]
        )
        self.assertTrue(np.all(data == self.combined_ex.ts))

    def test_data_size(self):
        self.assertEqual(130, self.combined_ex.ts.size)

    def test_has_data(self):
        self.assertEqual(True, self.combined_ex.has_data())


# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
