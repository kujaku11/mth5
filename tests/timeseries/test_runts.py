# -*- coding: utf-8 -*-
"""
Test time series

Created on Tue Jun 30 16:38:27 2020

:copyright:
    author: Jared Peacock
    
:license:
    MIT
    
"""
# =============================================================================
# imports
# =============================================================================

import unittest
import numpy as np

from mth5.timeseries import ChannelTS, RunTS

from mt_metadata.utils.mttime import MTime
import mt_metadata.timeseries as metadata
from mt_metadata.timeseries.filters import (
    PoleZeroFilter,
    ChannelResponseFilter,
)

# =============================================================================
# test run
# =============================================================================


class TestRunTSClass(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.run_object = RunTS()

        self.start = "2015-01-08T19:49:18+00:00"
        self.end = "2015-01-08T19:57:49.875000"
        self.sample_rate = 8
        self.npts = 4096

        self.ex = ChannelTS(
            "electric",
            data=np.random.rand(self.npts),
            channel_metadata={
                "electric": {
                    "component": "Ex",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
        )

        self.run_object.set_dataset([self.ex])

    def test_copy(self):
        run_copy = self.run_object.copy()

        self.assertEqual(self.run_object, run_copy)

    def test_set_run_metadata_fail(self):
        self.assertRaises(TypeError, RunTS, [self.ex], **{"run_metadata": []})

    def test_set_station_metadata_fail(self):
        self.assertRaises(
            TypeError, RunTS, [self.ex], **{"station_metadata": []}
        )

    def test_validate_run_metadata(self):
        self.assertEqual(
            self.run_object.run_metadata,
            self.run_object._validate_run_metadata(
                self.run_object.run_metadata
            ),
        )

    def test_validate_run_metadata_from_dict(self):
        self.assertEqual(
            metadata.Run(id="0"),
            self.run_object._validate_run_metadata({"id": "0"}),
        )

    def test_validate_station_metadata(self):
        self.assertEqual(
            self.run_object.station_metadata,
            self.run_object._validate_station_metadata(
                self.run_object.station_metadata
            ),
        )

    def test_validate_station_metadata_from_dict(self):
        self.assertEqual(
            metadata.Station(id="0"),
            self.run_object._validate_station_metadata({"id": "0"}),
        )

    def test_validate_survey_metadata(self):
        self.assertEqual(
            self.run_object.survey_metadata,
            self.run_object._validate_survey_metadata(
                self.run_object.survey_metadata
            ),
        )

    def test_validate_survey_metadata_from_dict(self):
        self.assertEqual(
            metadata.Survey(id="0"),
            self.run_object._validate_survey_metadata({"id": "0"}),
        )

    def test_validate_array_fail(self):
        with self.subTest("bad type"):
            self.assertRaises(
                TypeError, self.run_object._validate_array_list, 10
            )

        with self.subTest("bad list"):
            self.assertRaises(
                TypeError, self.run_object._validate_array_list, [10]
            )

    def test_sr_fail(self):
        hz = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hz",
                    "sample_rate": 1,
                    "time_period.start": self.start,
                }
            },
        )

        self.assertRaises(
            ValueError,
            self.run_object.set_dataset,
            [self.ex, hz],
        )

    def test_wrong_metadata(self):
        self.run_object.run_metadata.sample_rate = 10
        self.run_object.validate_metadata()

        with self.subTest("sample rate"):
            self.assertEqual(
                self.ex.sample_rate, self.run_object.run_metadata.sample_rate
            )
        with self.subTest("start"):
            self.run_object.run_metadata.start = "2020-01-01T00:00:00"
            self.run_object.validate_metadata()
            self.assertEqual(
                self.run_object.start,
                self.run_object.run_metadata.time_period.start,
            )
        with self.subTest("end"):
            self.run_object.run_metadata.end = "2020-01-01T00:00:00"
            self.run_object.validate_metadata()
            self.assertEqual(
                self.run_object.end,
                self.run_object.run_metadata.time_period.end,
            )

    def test_add_channel_xarray(self):
        x = self.ex.to_xarray()
        x.attrs["component"] = "ez"
        x.name = "ez"
        self.run_object.add_channel(x)

        self.assertEquals(
            sorted(self.run_object.channels),
            sorted(["ex", "ez"]),
        )


class TestMakeRunTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.maxDiff = None
        channel_list = []
        self.common_start = "2020-01-01T00:00:00+00:00"
        self.sample_rate = 1.0
        self.n_samples = 4096
        t = np.arange(self.n_samples)
        data = np.sum(
            [
                np.cos(2 * np.pi * w * t + phi)
                for w, phi in zip(np.logspace(-3, 3, 20), np.random.rand(20))
            ],
            axis=0,
        )

        self.station_metadata = metadata.Station(id="mt001")
        self.run_metadata = metadata.Run(id="001")

        for component in ["hx", "hy", "hz"]:
            h_metadata = metadata.Magnetic(component=component)
            h_metadata.time_period.start = self.common_start
            h_metadata.sample_rate = self.sample_rate
            h_channel = ChannelTS(
                channel_type="magnetic",
                data=data,
                channel_metadata=h_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
            channel_list.append(h_channel)

        for component in ["ex", "ey"]:
            e_metadata = metadata.Electric(component=component)
            e_metadata.time_period.start = self.common_start
            e_metadata.sample_rate = self.sample_rate
            e_channel = ChannelTS(
                channel_type="electric",
                data=data,
                channel_metadata=e_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
            channel_list.append(e_channel)

        aux_metadata = metadata.Auxiliary(component="temperature")
        aux_metadata.time_period.start = self.common_start
        aux_metadata.sample_rate = self.sample_rate
        aux_channel = ChannelTS(
            channel_type="auxiliary",
            data=np.random.rand(self.n_samples) * 30,
            channel_metadata=aux_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
        )
        channel_list.append(aux_channel)

        self.run_ts = RunTS(channel_list)

    def test_station_metadata(self):
        with self.subTest("station id"):
            self.assertEqual(
                self.run_ts.station_metadata.id, self.station_metadata.id
            )
        with self.subTest("start"):
            self.assertEqual(
                self.run_ts.station_metadata.time_period.start,
                self.common_start,
            )
        with self.subTest("run list"):
            self.assertListEqual(
                self.run_ts.station_metadata.run_list,
                [self.run_metadata.id],
            )
        with self.subTest("channels_recorded"):
            self.assertListEqual(
                ["ex", "ey", "hx", "hy", "hz", "temperature"],
                self.run_ts.station_metadata.channels_recorded,
            )

    def test_run_metadata(self):
        with self.subTest("run id"):
            self.assertEqual(self.run_ts.run_metadata.id, self.run_metadata.id)
        with self.subTest("start"):
            self.assertEqual(
                self.run_ts.run_metadata.time_period.start,
                self.common_start,
            )
        with self.subTest("channels"):
            self.assertListEqual(
                self.run_ts.run_metadata.channels_recorded_all,
                ["hx", "hy", "hz", "ex", "ey", "temperature"],
            )

    def test_channels(self):
        for comp in self.run_ts.channels:
            ch = getattr(self.run_ts, comp)
            with self.subTest("start"):
                self.assertEqual(
                    ch.channel_metadata.time_period.start,
                    self.common_start,
                )
            with self.subTest("sample rate"):
                self.assertEqual(
                    ch.sample_rate,
                    self.sample_rate,
                )
            with self.subTest("n samples"):
                self.assertEqual(
                    ch.n_samples,
                    4096,
                )


class TestRunTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        self.run_object = RunTS()
        self.maxDiff = None
        self.start = "2015-01-08T19:49:18+00:00"
        self.end = "2015-01-08T19:57:49.875000"
        self.sample_rate = 8
        self.npts = 4096

        pz = PoleZeroFilter(
            units_in="volts", units_out="nanotesla", name="instrument_response"
        )
        pz.poles = [
            (-6.283185 + 10.882477j),
            (-6.283185 - 10.882477j),
            (-12.566371 + 0j),
        ]
        pz.zeros = []
        pz.normalization_factor = 18244400

        self.cr = ChannelResponseFilter(filters_list=[pz])

        self.ex = ChannelTS(
            "electric",
            data=np.random.rand(self.npts),
            channel_metadata={
                "electric": {
                    "component": "Ex",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
            channel_response_filter=self.cr,
        )
        self.ey = ChannelTS(
            "electric",
            data=np.random.rand(self.npts),
            channel_metadata={
                "electric": {
                    "component": "Ey",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
            channel_response_filter=self.cr,
        )
        self.hx = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hx",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
            channel_response_filter=self.cr,
        )
        self.hy = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hy",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
            channel_response_filter=self.cr,
        )
        self.hz = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hz",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start,
                }
            },
            channel_response_filter=self.cr,
        )

        self.run_object.set_dataset(
            [self.ex, self.ey, self.hx, self.hy, self.hz]
        )

    def test_str(self):
        s_list = [
            f"Survey:      {self.run_object.survey_metadata.id}",
            f"Station:     {self.run_object.station_metadata.id}",
            f"Run:         {self.run_object.run_metadata.id}",
            f"Start:       {self.run_object.start}",
            f"End:         {self.run_object.end}",
            f"Sample Rate: {self.run_object.sample_rate}",
            f"Components:  {self.run_object.channels}",
        ]
        test_str = "\n\t".join(["RunTS Summary:"] + s_list)

        self.assertEqual(test_str, self.run_object.__str__())

    def test_repr(self):
        s_list = [
            f"Survey:      {self.run_object.survey_metadata.id}",
            f"Station:     {self.run_object.station_metadata.id}",
            f"Run:         {self.run_object.run_metadata.id}",
            f"Start:       {self.run_object.start}",
            f"End:         {self.run_object.end}",
            f"Sample Rate: {self.run_object.sample_rate}",
            f"Components:  {self.run_object.channels}",
        ]
        test_str = "\n\t".join(["RunTS Summary:"] + s_list)

        self.assertEqual(test_str, self.run_object.__repr__())

    def test_validate_run_metadata(self):
        self.assertEqual(
            self.run_object.run_metadata,
            self.run_object._validate_run_metadata(
                self.run_object.run_metadata
            ),
        )

    def test_validate_station_metadata(self):
        self.assertEqual(
            self.run_object.station_metadata,
            self.run_object._validate_station_metadata(
                self.run_object.station_metadata
            ),
        )

    def test_validate_survey_metadata(self):
        self.assertEqual(
            self.run_object.survey_metadata,
            self.run_object._validate_survey_metadata(
                self.run_object.survey_metadata
            ),
        )

    def test_initialize(self):

        with self.subTest("channels"):
            self.assertListEqual(
                ["ex", "ey", "hx", "hy", "hz"], self.run_object.channels
            )
        with self.subTest("sample rate"):
            self.assertEqual(self.run_object.sample_rate, self.sample_rate)
        with self.subTest("start"):
            self.assertEqual(self.run_object.start, MTime(self.start))
        with self.subTest("end"):
            self.assertEqual(self.run_object.end, MTime(self.end))

    def test_sample_interval(self):
        self.assertEqual(
            1.0 / self.sample_rate, self.run_object.sample_interval
        )

    def test_channels(self):

        for comp in ["ex", "ey", "hx", "hy", "hz"]:
            ch = getattr(self.run_object, comp)

            with self.subTest(msg=f"{comp} isinstance channel"):
                self.assertIsInstance(ch, ChannelTS)
            with self.subTest(msg=f"{comp} sample rate"):
                self.assertEqual(ch.sample_rate, self.sample_rate)
            with self.subTest(msg=f"{comp} start"):
                self.assertEqual(ch.start, MTime(self.start))
            with self.subTest(msg=f"{comp} end"):
                self.assertEqual(ch.end, MTime(self.end))
            with self.subTest(msg=f"{comp} component"):
                self.assertEqual(ch.component, comp)

            with self.subTest(msg=f"{comp} filters"):
                self.assertListEqual(
                    self.cr.filters_list,
                    ch.channel_response_filter.filters_list,
                )

    def test_get_channel_fail(self):
        """
        self.run_object.temperature should return None, because 'temperature' is not in self.channels
        :return:
        """

        self.assertRaises(NameError, getattr, *(self.run_object, "temperature"))

    def test_get_slice(self):

        start = "2015-01-08T19:49:30+00:00"
        npts = 256

        r_slice = self.run_object.get_slice(start, n_samples=npts)

        with self.subTest("isinstance runts"):
            self.assertIsInstance(r_slice, RunTS)
        with self.subTest("sample rate"):
            self.assertEqual(r_slice.sample_rate, self.sample_rate)
        with self.subTest("start not equal"):
            self.assertEqual(r_slice.start, MTime(start))

        with self.subTest("start equal"):
            # the time index does not have a value at the requested location
            # so it grabs the closest one.
            self.assertEqual(r_slice.start, MTime(start))
        with self.subTest("end"):
            self.assertEqual(
                r_slice.end, MTime("2015-01-08T19:50:01.875000+00:00")
            )

        with self.subTest("npts"):
            self.assertEqual(r_slice.dataset.ex.data.shape[0], npts)

    def test_filters_dict(self):
        self.assertEqual(
            list(self.run_object.filters.keys()), ["instrument_response"]
        )

    def test_filters_fail(self):
        def set_filters(value):
            self.run_object.filters = value

        self.assertRaises(TypeError, set_filters, ())

    def test_summarize_metadata(self):
        meta_dict = {}
        for comp in self.run_object.dataset.data_vars:
            for mkey, mvalue in self.run_object.dataset[comp].attrs.items():
                meta_dict[f"{comp}.{mkey}"] = mvalue
        self.assertDictEqual(meta_dict, self.run_object.summarize_metadata)

    def test_to_obspy_stream(self):
        stream = self.run_object.to_obspy_stream()

        with self.subTest("count"):
            self.assertEqual(stream.count(), 5)

        for tr in stream.traces:
            with self.subTest("sample_rate"):
                self.assertEqual(tr.stats.sampling_rate, self.sample_rate)

            with self.subTest("start time"):
                self.assertEqual(tr.stats.starttime, self.start)

            with self.subTest("npts"):
                self.assertEqual(tr.stats.npts, self.npts)


class TestMergeRunTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(0)
        self.maxDiff = None
        self.sample_rate = 8
        self.npts = 4096
        self.start_01 = "2015-01-08T19:49:18+00:00"
        self.end_01 = "2015-01-08T19:57:49.875000"
        self.start_02 = "2015-01-08T19:57:52+00:00"
        self.end_02 = "2015-01-08T20:06:23.875000+00:00"

        self.pz1 = PoleZeroFilter(
            units_in="volts", units_out="nanotesla", name="filter_1"
        )
        self.pz1.poles = [
            (-6.283185 + 10.882477j),
            (-6.283185 - 10.882477j),
            (-12.566371 + 0j),
        ]
        self.pz1.zeros = []
        self.pz1.normalization_factor = 18244400
        self.pz2 = self.pz1.copy()
        self.pz2.name = "filter_2"

        self.cr_01 = ChannelResponseFilter(filters_list=[self.pz1])
        self.cr_02 = ChannelResponseFilter(filters_list=[self.pz2])

        self.run_object_01 = RunTS()
        self.ey_01 = ChannelTS(
            "electric",
            data=np.random.rand(self.npts),
            channel_metadata={
                "electric": {
                    "component": "Ey",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start_01,
                }
            },
            channel_response_filter=self.cr_01,
        )
        self.hx_01 = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hx",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start_01,
                }
            },
            channel_response_filter=self.cr_01,
        )

        self.run_object_01.set_dataset([self.ey_01, self.hx_01])

        self.run_object_02 = RunTS()
        self.ey_02 = ChannelTS(
            "electric",
            data=np.random.rand(self.npts),
            channel_metadata={
                "electric": {
                    "component": "Ey",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start_02,
                }
            },
            channel_response_filter=self.cr_02,
        )
        self.hx_02 = ChannelTS(
            "magnetic",
            data=np.random.rand(self.npts),
            channel_metadata={
                "magnetic": {
                    "component": "hx",
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start_02,
                }
            },
            channel_response_filter=self.cr_02,
        )

        self.run_object_02.set_dataset([self.ey_02, self.hx_02])

        self.combined_run = self.run_object_01 + self.run_object_02
        self.merged_run = self.run_object_01.merge(self.run_object_02)
        self.merged_run_sr01 = self.run_object_01.merge(
            self.run_object_02, new_sample_rate=1
        )

    def test_add_runs(self):
        with self.subTest("size"):
            self.assertEqual(
                self.combined_run.dataset.sizes["time"], 2 * self.npts + 16
            )

        with self.subTest("start"):
            self.assertEqual(self.combined_run.start, self.start_01)

        with self.subTest("end"):
            self.assertEqual(self.combined_run.end, self.end_02)

        with self.subTest("filters"):
            self.assertDictEqual(
                self.combined_run.filters,
                {self.pz1.name: self.pz1, self.pz2.name: self.pz2},
            )

        with self.subTest("run.start"):
            self.assertEqual(
                self.combined_run.run_metadata.time_period.start, self.start_01
            )

        with self.subTest("run.end"):
            self.assertEqual(
                self.combined_run.run_metadata.time_period.end, self.end_02
            )
        with self.subTest("station.start"):
            self.assertEqual(
                self.combined_run.station_metadata.time_period.start,
                self.start_01,
            )

        with self.subTest("station.end"):
            self.assertEqual(
                self.combined_run.station_metadata.time_period.end, self.end_02
            )

        with self.subTest("run.sample_rate"):
            self.assertEqual(
                self.combined_run.run_metadata.sample_rate, self.sample_rate
            )

        with self.subTest("channels"):
            self.assertListEqual(
                ["ey", "hx"],
                self.combined_run.channels,
            )

    def test_merge_runs(self):
        with self.subTest("size"):
            self.assertEqual(
                self.merged_run.dataset.sizes["time"], 2 * self.npts + 16
            )

        with self.subTest("start"):
            self.assertEqual(self.merged_run.start, self.start_01)

        with self.subTest("end"):
            self.assertEqual(self.merged_run.end, self.end_02)

        with self.subTest("filters"):
            self.assertDictEqual(
                self.merged_run.filters,
                {self.pz1.name: self.pz1, self.pz2.name: self.pz2},
            )

        with self.subTest("run.start"):
            self.assertEqual(
                self.merged_run.run_metadata.time_period.start, self.start_01
            )

        with self.subTest("run.end"):
            self.assertEqual(
                self.merged_run.run_metadata.time_period.end, self.end_02
            )
        with self.subTest("station.start"):
            self.assertEqual(
                self.merged_run.station_metadata.time_period.start,
                self.start_01,
            )

        with self.subTest("station.end"):
            self.assertEqual(
                self.merged_run.station_metadata.time_period.end, self.end_02
            )

        with self.subTest("run.sample_rate"):
            self.assertEqual(
                self.merged_run.run_metadata.sample_rate, self.sample_rate
            )

        with self.subTest("channels"):
            self.assertListEqual(
                ["ey", "hx"],
                self.merged_run.channels,
            )

    def test_merge_runs_decimated(self):
        with self.subTest("size"):
            self.assertEqual(
                self.merged_run_sr01.dataset.sizes["time"],
                (2 * self.npts + 16) / 8,
            )

        with self.subTest("start"):
            self.assertEqual(self.merged_run_sr01.start, self.start_01)

        with self.subTest("end"):
            self.assertEqual(
                self.merged_run_sr01.end, "2015-01-08T20:06:23+00:00"
            )

        with self.subTest("filters"):
            self.assertDictEqual(
                self.merged_run_sr01.filters,
                {self.pz1.name: self.pz1, self.pz2.name: self.pz2},
            )

        with self.subTest("run.start"):
            self.assertEqual(
                self.merged_run_sr01.run_metadata.time_period.start,
                self.start_01,
            )

        with self.subTest("run.end"):
            self.assertEqual(
                self.merged_run_sr01.run_metadata.time_period.end,
                "2015-01-08T20:06:23+00:00",
            )
        with self.subTest("station.start"):
            self.assertEqual(
                self.merged_run_sr01.station_metadata.time_period.start,
                self.start_01,
            )

        with self.subTest("station.end"):
            self.assertEqual(
                self.merged_run_sr01.station_metadata.time_period.end,
                "2015-01-08T20:06:23+00:00",
            )

        with self.subTest("run.sample_rate"):
            self.assertEqual(self.merged_run_sr01.run_metadata.sample_rate, 1)

        with self.subTest("channels"):
            self.assertListEqual(
                ["ey", "hx"],
                self.merged_run_sr01.channels,
            )


class TestMisalignedRuns(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.maxDiff = None
        channel_list = []
        self.common_start = "2020-01-01T00:00:00+00:00"
        self.sample_rate = 1.0
        self.hx_n_samples = 4098
        self.ey_n_samples = 4096
        self.station_metadata = metadata.Station(id="mt001")
        self.run_metadata = metadata.Run(id="001")
        channel_list = []

        ### HX
        hx_metadata = metadata.Magnetic(component="hx")
        hx_metadata.time_period.start = self.common_start
        hx_metadata.sample_rate = self.sample_rate

        t = np.arange(self.hx_n_samples)
        data = np.sum(
            [
                np.cos(2 * np.pi * w * t + phi)
                for w, phi in zip(np.logspace(-3, 3, 20), np.random.rand(20))
            ],
            axis=0,
        )

        self.hx = ChannelTS(
            channel_type="magnetic",
            data=data,
            channel_metadata=hx_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
        )
        channel_list.append(self.hx)

        ## EY
        ey_metadata = metadata.Electric(component="ey")
        ey_metadata.time_period.start = self.common_start
        ey_metadata.sample_rate = self.sample_rate

        t = np.arange(self.ey_n_samples)
        data = np.sum(
            [
                np.cos(2 * np.pi * w * t + phi)
                for w, phi in zip(np.logspace(-3, 3, 20), np.random.rand(20))
            ],
            axis=0,
        )

        self.ey = ChannelTS(
            channel_type="electric",
            data=data,
            channel_metadata=ey_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
        )
        channel_list.append(self.ey)

        ## bad channel
        ex_metadata = metadata.Electric(component="ex")
        ex_metadata.time_period.start = "2021-05-05T12:10:05+00:00"
        ex_metadata.sample_rate = self.sample_rate + 10

        self.bad_ch = ChannelTS(
            channel_type="electric",
            data=data,
            channel_metadata=ex_metadata,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
        )

        self.run_ts = RunTS(channel_list)
        self.ch_list = [self.ey.data_array, self.hx.data_array]
        self.bad_ch_list = [
            self.ey.data_array,
            self.hx.data_array,
            self.bad_ch.data_array,
        ]

    def test_check_sample_rate(self):
        self.assertEqual(
            self.sample_rate,
            self.run_ts._check_sample_rate(self.ch_list),
        )

    def test_check_sample_rate_fail(self):
        self.assertRaises(
            ValueError,
            self.run_ts._check_sample_rate,
            **{"valid_list": self.bad_ch_list},
        )

    def test_common_start(self):
        self.assertEqual(True, self.run_ts._check_common_start(self.ch_list))

    def test_common_start_fail(self):
        self.assertEqual(
            False, self.run_ts._check_common_start(self.bad_ch_list)
        )

    def test_common_end(self):
        self.assertEqual(False, self.run_ts._check_common_end(self.ch_list))

    def test_common_end_fail(self):
        self.assertEqual(False, self.run_ts._check_common_end(self.bad_ch_list))

    def test_earliest_start(self):
        self.assertEqual(
            self.ey.data_array.coords["time"].values[0],
            self.run_ts._get_earliest_start(self.ch_list),
        )

    def test_latest_end(self):
        self.assertEqual(
            self.hx.data_array.coords["time"].values[-1],
            self.run_ts._get_latest_end(self.ch_list),
        )

    def test_get_common_time_index(self):
        dt = self.run_ts._get_common_time_index(
            self.ey.data_array.coords["time"].values[0],
            self.hx.data_array.coords["time"].values[-1],
            self.sample_rate,
        )

        earliest_start = self.run_ts._get_earliest_start(self.ch_list)
        latest_end = self.run_ts._get_latest_end(self.ch_list)

        dt2 = self.run_ts._get_common_time_index(
            earliest_start, latest_end, self.sample_rate
        )

        self.assertTrue((dt == dt2).all())

    def test_run_start(self):
        self.assertEqual(self.run_ts.start, self.common_start)

    def test_run_end(self):
        self.assertEqual(self.run_ts.end, self.hx.end)

    def test_run_nsamples(self):
        self.assertEqual(self.run_ts.dataset.sizes["time"], self.hx_n_samples)

    def test_run_sample_rate(self):
        self.assertEqual(self.run_ts.sample_rate, self.sample_rate)


# =============================================================================
# run tests
# =============================================================================
if __name__ == "__main__":
    unittest.main()
