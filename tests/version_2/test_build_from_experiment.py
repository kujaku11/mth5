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
from mth5.helpers import validate_name
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
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = mth5.MTH5(file_version="0.2.0")
        self.mth5_obj.open_mth5(self.fn, mode="w")
        self.experiment = Experiment()
        self.experiment.from_xml(fn=MT_EXPERIMENT_SINGLE_STATION)
        self.survey_name = validate_name(self.experiment.surveys[0].id)
        self.mth5_obj.from_experiment(self.experiment)

    def test_surveys(self):
        survey = self.experiment.surveys[0]
        sg = self.mth5_obj.get_survey(self.survey_name)

        sd_md = survey.to_dict(single=True)
        sd_md.pop("hdf5_reference")
        sd_md.pop("mth5_type")

        h5_md = sg.metadata.to_dict(single=True)
        h5_md.pop("hdf5_reference")
        h5_md.pop("mth5_type")

        self.assertDictEqual(sd_md, h5_md)

    def test_stations(self):
        stations = self.experiment.surveys[0].stations
        for station in stations:
            with self.subTest(name=station.id):
                h5_station = self.mth5_obj.get_station(station.id, self.survey_name)
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
            with self.subTest(name=run.id):
                h5_run = self.mth5_obj.get_run(
                    self.experiment.surveys[0].stations[0].id,
                    run.id,
                    survey=self.survey_name,
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
            with self.subTest(name=run.id):
                h5_run = self.mth5_obj.get_run(
                    self.experiment.surveys[0].stations[0].id,
                    run.id,
                    self.survey_name,
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
        sg = self.mth5_obj.get_survey(self.survey_name)

        for key, value in exp_filters.items():
            with self.subTest(name=key):
                key = key.replace("/", " per ").lower()
                sd = value.to_dict(single=True, required=False)
                h5_sd = sg.filters_group.to_filter_object(key)
                h5_sd = h5_sd.to_dict(single=True, required=False)
                for k in sd.keys():
                    with self.subTest(f"{key}_{k}"):
                        v1 = sd[k]
                        v2 = h5_sd[k]
                        if isinstance(v1, (float, int)):
                            self.assertAlmostEqual(v1, float(v2), 5)
                        elif isinstance(v1, np.ndarray):
                            self.assertEqual(v1.dtype, v2.dtype)
                            self.assertTrue((v1==v2).all())
                        else:
                            self.assertEqual(v1, v2)

            # self.assertDictEqual(h5_sd, sd)

    def tearDown(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()
