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

from mth5 import mth5
from mth5.utils.exceptions import MTH5Error
from mth5.timeseries import ChannelTS, RunTS
from mth5.groups.standards import summarize_metadata_standards
from mt_metadata.utils.mttime import MTime
from mt_metadata.timeseries import Experiment
from mt_metadata.utils import MT_EXPERIMENT_SINGLE_STATION

fn_path = Path(__file__).parent
# =============================================================================
#
# =============================================================================
mth5.helpers.close_open_files()


class TestMTH5(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = mth5.MTH5()
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
                v1 = sd[k]
                v2 = h5_sd[k]
                if isinstance(v1, (float, int)):
                    self.assertAlmostEqual(v1, float(v2), 5)
                else:
                    self.assertEqual(v1, v2)

            # self.assertDictEqual(h5_sd, sd)

    def tearDown(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()
