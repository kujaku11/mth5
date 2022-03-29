# -*- coding: utf-8 -*-
"""
Created on Thu May 13 13:45:27 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================

import unittest
from pathlib import Path
import numpy as np

from mth5 import mth5, CHANNEL_DTYPE
from mt_metadata.timeseries import Experiment
from mt_metadata import MT_EXPERIMENT_SINGLE_STATION

fn_path = Path(__file__).parent
# =============================================================================
#
# =============================================================================
mth5.helpers.close_open_files()


class TestMTH5(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.fn = fn_path.joinpath("test.h5")
        self.mth5_obj = mth5.MTH5(file_version="0.1.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        self.experiment = Experiment()
        self.experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        self.mth5_obj.from_experiment(self.experiment)

    def test_surveys(self):
        survey = self.experiment.surveys[0]
        self.assertEqual(survey, self.mth5_obj.survey_group.metadata)

    def test_stations(self):
        stations = self.experiment.surveys[0].stations
        for station in stations:
            with self.subTest(station.id):
                h5_station = self.mth5_obj.get_station(station.id)
                sd = station.to_dict(single=True)
                sd.pop("hdf5_reference")
                sd.pop("mth5_type")

                h5_sd = h5_station.metadata.to_dict(single=True)
                h5_sd.pop("hdf5_reference")
                h5_sd.pop("mth5_type")

                self.assertDictEqual(h5_sd, sd)

    def test_runs(self):
        runs = self.experiment.surveys[0].stations[0].runs
        for run in runs:
            with self.subTest(run.id):
                h5_run = self.mth5_obj.get_run(
                    self.experiment.surveys[0].stations[0].id, run.id
                )
                sd = run.to_dict(single=True)
                sd.pop("hdf5_reference")
                sd.pop("mth5_type")

                h5_sd = h5_run.metadata.to_dict(single=True)
                h5_sd.pop("hdf5_reference")
                h5_sd.pop("mth5_type")

                self.assertDictEqual(h5_sd, sd)

    def test_channels(self):
        runs = self.experiment.surveys[0].stations[0].runs
        for run in runs:
            h5_run = self.mth5_obj.get_run(
                self.experiment.surveys[0].stations[0].id, run.id
            )
            for channel in run.channels:
                with self.subTest(f"{run.id}/ch.component"):
                    h5_channel = h5_run.get_channel(channel.component)

                    sd = channel.to_dict(single=True)
                    sd.pop("hdf5_reference")
                    sd.pop("mth5_type")

                    h5_sd = h5_channel.metadata.to_dict(single=True)
                    h5_sd.pop("hdf5_reference")
                    h5_sd.pop("mth5_type")

                    self.assertDictEqual(h5_sd, sd)

    def test_filters(self):
        exp_filters = self.experiment.surveys[0].filters

        for key, value in exp_filters.items():
            key = key.replace("/", " per ").lower()
            sd = value.to_dict(single=True, required=False)
            h5_sd = self.mth5_obj.filters_group.to_filter_object(key)
            h5_sd = h5_sd.to_dict(single=True, required=False)
            for k in sd.keys():
                with self.subTest(f"{key}_{k}"):
                    v1 = sd[k]
                    v2 = h5_sd[k]
                    if isinstance(v1, (float, int)):
                        self.assertAlmostEqual(v1, float(v2), 5)
                    elif isinstance(v1, np.ndarray):
                        self.assertEqual(v1.dtype, v2.dtype)
                        self.assertTrue((v1 == v2).all())
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

    def tearDown(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()
        
class TestUpdateFromExperiment(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.fn = fn_path.joinpath("test.h5")
        self.mth5_obj = mth5.MTH5(file_version="0.1.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        self.experiment = Experiment()
        self.experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        self.mth5_obj.from_experiment(self.experiment)
        
        self.experiment_02 = Experiment()
        self.experiment_02.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        self.experiment_02.surveys[0].id = "different_survey_name"
        self.experiment_02.surveys[0].stations[0].location.latitude = 10
        
        
    def test_update_from_new_experiment(self):
        
        self.mth5_obj.from_experiment(self.experiment_02, update=True)
        
        with self.subTest("new_survey"):
            self.assertEqual(self.mth5_obj.survey_group.metadata.id,
                             self.experiment_02.surveys[0].id)
            
        with self.subTest("new_location"):
            st = self.mth5_obj.get_station("REW09")
            self.assertEqual(
                st.metadata.location.latitude,
                self.experiment_02.surveys[0].stations[0].location.latitude)
            
    def tearDown(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()
            
