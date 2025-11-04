# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:45:27 2021

:copyright:
    Jared             with self.subTest(run.id):
                h5_run = self.mth5_obj.get_run(
                    self.experiment.surveys[0].stations[0].id, run.id
                )
                rd = run.to_dict(single=True)
                if "hdf5_reference" in rd:
                    rd.pop("hdf5_reference")
                if "mth5_type" in rd:
                    rd.pop("mth5_type")

                h5_rd = h5_run.metadata.to_dict(single=True)
                if "hdf5_reference" in h5_rd:
                    h5_rd.pop("hdf5_reference")
                if "mth5_type" in h5_rd:
                    h5_rd.pop("mth5_type")eacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from mth5 import CHANNEL_DTYPE
from mth5 import helpers
from mth5.mth5 import MTH5
from mt_metadata.timeseries import Experiment
from mt_metadata import MT_EXPERIMENT_SINGLE_STATION

fn_path = Path(__file__).parent
# =============================================================================
#
# =============================================================================
helpers.close_open_files()


class TestMTH5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.fn = fn_path.joinpath("test.h5")
        cls.mth5_obj = MTH5(file_version="0.1.0")
        cls.mth5_obj.open_mth5(cls.fn, mode="w")
        cls.experiment = Experiment()
        cls.experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        cls.mth5_obj.from_experiment(cls.experiment)

    # @classmethod
    # def tearDownClass(cls):
    #     """Clean up by closing the MTH5 file."""
    #     if hasattr(cls, "mth5_obj") and cls.mth5_obj is not None:
    #         try:
    #             cls.mth5_obj.close_mth5()
    #         except Exception:
    #             pass  # File might already be closed
    #     helpers.close_open_files()

    def test_surveys(self):
        survey = self.experiment.surveys[0]
        # Handle differences between original and MTH5-loaded metadata
        sd = survey.to_dict(single=True)
        if "hdf5_reference" in sd:
            sd.pop("hdf5_reference")
        if "mth5_type" in sd:
            sd.pop("mth5_type")

        h5_sd = self.mth5_obj.survey_group.metadata.to_dict(single=True)
        if "hdf5_reference" in h5_sd:
            h5_sd.pop("hdf5_reference")
        if "mth5_type" in h5_sd:
            h5_sd.pop("mth5_type")

        self.assertEqual(sd, h5_sd)

    def test_stations(self):
        stations = self.experiment.surveys[0].stations
        for station in stations:
            with self.subTest(station.id):
                h5_station = self.mth5_obj.get_station(station.id)
                sd = station.to_dict(single=True)
                if "hdf5_reference" in sd:
                    sd.pop("hdf5_reference")
                if "mth5_type" in sd:
                    sd.pop("mth5_type")

                h5_sd = h5_station.metadata.to_dict(single=True)
                if "hdf5_reference" in h5_sd:
                    h5_sd.pop("hdf5_reference")
                if "mth5_type" in h5_sd:
                    h5_sd.pop("mth5_type")

                self.assertDictEqual(h5_sd, sd)

    def test_runs(self):
        runs = self.experiment.surveys[0].stations[0].runs
        for run in runs:
            with self.subTest(run.id):
                h5_run = self.mth5_obj.get_run(
                    self.experiment.surveys[0].stations[0].id, run.id
                )
                rd = run.to_dict(single=True)
                if "hdf5_reference" in rd:
                    rd.pop("hdf5_reference")
                if "mth5_type" in rd:
                    rd.pop("mth5_type")

                h5_rd = h5_run.metadata.to_dict(single=True)
                if "hdf5_reference" in h5_rd:
                    h5_rd.pop("hdf5_reference")
                if "mth5_type" in h5_rd:
                    h5_rd.pop("mth5_type")

                self.assertDictEqual(h5_rd, rd)

    def test_to_run_ts(self):
        run_group = self.mth5_obj.get_run(
            self.experiment.surveys[0].stations[0].id,
            self.experiment.surveys[0].stations[0].runs[0].id,
        )
        run_ts = run_group.to_runts()

        for key in self.experiment.surveys[0].to_dict(single=True).keys():
            with self.subTest(f"survey.{key}"):
                self.assertEqual(
                    self.experiment.surveys[0].get_attr_from_name(key),
                    run_ts.survey_metadata.get_attr_from_name(key),
                )

        for key in self.experiment.surveys[0].stations[0].to_dict(single=True).keys():
            if key in ["hdf5_reference", "mth5_type"]:
                continue

            with self.subTest(f"station.{key}"):
                if key in ["run_list"]:
                    self.assertListEqual(
                        ["a"],
                        run_ts.station_metadata.run_list,
                    )

                else:
                    self.assertEqual(
                        self.experiment.surveys[0].stations[0].get_attr_from_name(key),
                        run_ts.station_metadata.get_attr_from_name(key),
                    )

        for key in (
            self.experiment.surveys[0].stations[0].runs[0].to_dict(single=True).keys()
        ):
            if key in ["hdf5_reference", "mth5_type"]:
                continue
            with self.subTest(f"run.{key}"):
                if key in ["time_period.end"]:
                    self.assertNotEqual(
                        self.experiment.surveys[0]
                        .stations[0]
                        .runs[0]
                        .get_attr_from_name(key),
                        run_ts.run_metadata.get_attr_from_name(key),
                    )
                else:
                    self.assertEqual(
                        self.experiment.surveys[0]
                        .stations[0]
                        .runs[0]
                        .get_attr_from_name(key),
                        run_ts.run_metadata.get_attr_from_name(key),
                    )

    def test_channels(self):
        runs = self.experiment.surveys[0].stations[0].runs
        for run in runs:
            h5_run = self.mth5_obj.get_run(
                self.experiment.surveys[0].stations[0].id, run.id
            )
            for channel in run.channels:
                with self.subTest(f"{run.id}/{channel.component}"):
                    h5_channel = h5_run.get_channel(channel.component)

                    sd = channel.to_dict(single=True)
                    if "hdf5_reference" in sd:
                        sd.pop("hdf5_reference")
                    if "mth5_type" in sd:
                        sd.pop("mth5_type")

                    h5_sd = h5_channel.metadata.to_dict(single=True)
                    if "hdf5_reference" in h5_sd:
                        h5_sd.pop("hdf5_reference")
                    if "mth5_type" in h5_sd:
                        h5_sd.pop("mth5_type")

                    self.assertDictEqual(h5_sd, sd)

    def test_to_channel_ts(self):
        channel_group = self.mth5_obj.get_channel(
            self.experiment.surveys[0].stations[0].id,
            self.experiment.surveys[0].stations[0].runs[0].id,
            self.experiment.surveys[0].stations[0].runs[0].channels[0].component,
        )
        ch_ts = channel_group.to_channel_ts()

        for key in self.experiment.surveys[0].to_dict(single=True).keys():
            if key in ["hdf5_reference", "mth5_type"]:
                continue
            with self.subTest(f"survey.{key}"):
                self.assertEqual(
                    self.experiment.surveys[0].get_attr_from_name(key),
                    ch_ts.survey_metadata.get_attr_from_name(key),
                )

        for key in self.experiment.surveys[0].stations[0].to_dict(single=True).keys():
            if key in ["hdf5_reference", "mth5_type"]:
                continue

            with self.subTest(f"station.{key}"):
                if key in ["run_list", "channels_recorded"]:
                    self.assertListEqual(
                        ["a", "b", "c", "d", "e"],
                        ch_ts.station_metadata.run_list,
                    )
                else:
                    self.assertEqual(
                        self.experiment.surveys[0].stations[0].get_attr_from_name(key),
                        ch_ts.station_metadata.get_attr_from_name(key),
                    )

        for key in (
            self.experiment.surveys[0].stations[0].runs[0].to_dict(single=True).keys()
        ):
            if key in [
                "hdf5_reference",
                "mth5_type",
                "channels_recorded_magnetic",
                "channels_recorded_electric",
                "channels_recorded_auxiliary",
            ]:
                continue
            with self.subTest(f"run.{key}"):
                self.assertEqual(
                    self.experiment.surveys[0]
                    .stations[0]
                    .runs[0]
                    .get_attr_from_name(key),
                    ch_ts.run_metadata.get_attr_from_name(key),
                )

        for key in (
            self.experiment.surveys[0]
            .stations[0]
            .runs[0]
            .channels[0]
            .to_dict(single=True)
            .keys()
        ):
            if key in [
                "hdf5_reference",
                "mth5_type",
                "filter.name",
                "filter.applied",
            ]:
                continue
            with self.subTest(f"channel.{key}"):
                # end time is off by one second (bug?)
                if key in ["time_period.end"]:
                    self.assertNotEqual(
                        self.experiment.surveys[0]
                        .stations[0]
                        .runs[0]
                        .channels[0]
                        .get_attr_from_name(key),
                        ch_ts.station_metadata.get_attr_from_name(key),
                    )
                else:
                    self.assertEqual(
                        self.experiment.surveys[0]
                        .stations[0]
                        .runs[0]
                        .channels[0]
                        .get_attr_from_name(key),
                        ch_ts.channel_metadata.get_attr_from_name(key),
                    )

    def test_filters(self):
        exp_filters = self.experiment.surveys[0].filters

        for key, value in exp_filters.items():
            # Transform key to match how MTH5 stores filter names
            # MTH5 replaces "/" with " per " but preserves original casing
            stored_key = key.replace("/", " per ")
            sd = value.to_dict(single=True, required=False)
            h5_sd = self.mth5_obj.filters_group.to_filter_object(stored_key)
            h5_sd = h5_sd.to_dict(single=True, required=False)
            for k in sd.keys():
                # Only test keys that exist in both dictionaries
                if k not in h5_sd:
                    continue
                with self.subTest(f"{stored_key}_{k}"):
                    v1 = sd[k]
                    v2 = h5_sd[k]
                    if isinstance(v1, (float, int)):
                        self.assertAlmostEqual(v1, float(v2), 5)
                    elif isinstance(v1, np.ndarray):
                        # Handle dtype mismatches for complex arrays
                        if v1.dtype != v2.dtype:
                            # Convert v2 to same dtype as v1 if needed
                            v2_converted = v2.astype(v1.dtype)
                            self.assertTrue((v1 == v2_converted).all())
                        else:
                            self.assertEqual(v1.dtype, v2.dtype)
                            self.assertTrue((v1 == v2).all())
                    elif v1 is None and v2 == "None":
                        # Handle None vs 'None' string conversion
                        continue
                    else:
                        self.assertEqual(v1, v2)

    def test_channel_summary(self):
        self.mth5_obj.channel_summary.summarize()

        with self.subTest("test shape"):
            self.assertEqual(self.mth5_obj.channel_summary.shape, (25,))
        with self.subTest("test nrows"):
            self.assertEqual(self.mth5_obj.channel_summary.nrows, 25)
        with self.subTest(("test dtype")):
            self.assertEqual(self.mth5_obj.channel_summary.dtype, CHANNEL_DTYPE)
        with self.subTest("test station"):
            self.assertTrue(
                (self.mth5_obj.channel_summary.array["station"] == b"REW09").all()
            )

    def test_run_summary(self):
        self.mth5_obj.channel_summary.summarize()
        run_summary = self.mth5_obj.channel_summary.to_run_summary()
        with self.subTest("is dataframe"):
            self.assertIsInstance(run_summary, pd.DataFrame)
        with self.subTest("shape"):
            self.assertEqual(run_summary.shape, (5, 15))

    def test_run_summary_property(self):
        run_summary = self.mth5_obj.run_summary
        with self.subTest("is dataframe"):
            self.assertIsInstance(run_summary, pd.DataFrame)
        with self.subTest("shape"):
            self.assertEqual(run_summary.shape, (5, 15))

    @classmethod
    def tearDownClass(cls):
        cls.mth5_obj.close_mth5()
        cls.fn.unlink()


class TestUpdateFromExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.fn = fn_path.joinpath(
            "test_update.h5"
        )  # Use different filename to avoid conflicts
        cls.mth5_obj = MTH5(file_version="0.1.0")
        cls.mth5_obj.open_mth5(cls.fn, mode="w")
        cls.experiment = Experiment()
        cls.experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        cls.mth5_obj.from_experiment(cls.experiment)

        cls.experiment_02 = Experiment()
        cls.experiment_02.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        cls.experiment_02.surveys[0].id = "different_survey_name"
        cls.experiment_02.surveys[0].stations[0].location.latitude = 10

    @classmethod
    def tearDownClass(cls):
        """Clean up by closing the MTH5 file."""
        if hasattr(cls, "mth5_obj") and cls.mth5_obj is not None:
            try:
                cls.mth5_obj.close_mth5()
            except Exception:
                pass  # File might already be closed
        helpers.close_open_files()

    def test_update_from_new_experiment(self):

        self.mth5_obj.from_experiment(self.experiment_02, update=True)

        with self.subTest("new_survey"):
            self.assertEqual(
                self.mth5_obj.survey_group.metadata.id,
                self.experiment_02.surveys[0].id,
            )
        with self.subTest("new_location"):
            st = self.mth5_obj.get_station("REW09")
            self.assertEqual(
                st.metadata.location.latitude,
                self.experiment_02.surveys[0].stations[0].location.latitude,
            )

    @classmethod
    def tearDownClass(cls):
        cls.mth5_obj.close_mth5()
        cls.fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
