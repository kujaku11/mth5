# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:58:47 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

import unittest
from pathlib import Path
import numpy as np
import numpy.testing as npt
from mth5 import mth5

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mt_metadata import STATIONXML_FAP

fn_path = Path(__file__).parent


class TestFAPMTH5(unittest.TestCase):
    """
    Test making an MTH5 file from a FAP filtered StationXML

    """

    def setUp(self):
        self.translator = XMLInventoryMTExperiment()
        self.experiment = self.translator.xml_to_mt(stationxml_fn=STATIONXML_FAP)
        self.experiment.surveys[0].id = "test"
        self.base_path = "Experiment/Surveys/test"
        self.fn = fn_path.joinpath("from_fap_stationxml.h5")

        self.m = mth5.MTH5(file_version="0.2.0")
        self.m.open_mth5(self.fn, mode="a")
        self.m.from_experiment(self.experiment)

    def test_has_survey(self):
        self.assertEqual(self.m.has_group(self.base_path), True)

    def test_has_station(self):
        with self.subTest(name="stations group"):
            self.assertEqual(self.m.has_group(f"{self.base_path}/Stations"), True)
        with self.subTest(name="station fl001"):
            self.assertEqual(self.m.has_group(f"{self.base_path}/Stations/FL001"), True)

    def test_has_run_a(self):
        self.assertEqual(self.m.has_group(f"{self.base_path}/Stations/FL001/a"), True)

    def test_has_run_b(self):
        self.assertEqual(self.m.has_group(f"{self.base_path}/Stations/FL001/b"), True)

    def test_has_hx_a(self):
        self.assertEqual(
            self.m.has_group(f"{self.base_path}/Stations/FL001/a/hx"), True
        )

    def test_has_hx_b(self):
        self.assertEqual(
            self.m.has_group(f"{self.base_path}/Stations/FL001/b/hx"), True
        )

    def test_has_fap_table(self):

        self.assertEqual(
            self.m.has_group(
                f"{self.base_path}/Filters/fap/frequency response table_00"
            ),
            True,
        )

    def test_has_coefficient_filter(self):
        self.assertEqual(
            self.m.has_group(
                f"{self.base_path}/Filters/coefficient/v to counts (electric)"
            ),
            True,
        )

    def test_get_channel(self):
        self.hx = self.m.get_channel("FL001", "a", "hx", "test")
        fnames = [f.name for f in self.hx.channel_response_filter.filters_list]

        self.assertIn("frequency response table_00", fnames)
        self.assertIn("v to counts (electric)", fnames)

    def test_fap(self):
        self.hx = self.m.get_channel("FL001", "a", "hx", "test")
        fap = self.hx.channel_response_filter.filters_list[0]
        fap_exp = self.experiment.surveys[0].filters["frequency response table_00"]

        self.assertTrue(np.allclose(fap.frequencies, fap_exp.frequencies, 7))
        self.assertTrue(np.allclose(fap.amplitudes, fap_exp.amplitudes, 7))
        self.assertTrue(np.allclose(fap.phases, fap_exp.phases, 7))

        npt.assert_almost_equal(fap.frequencies, fap_exp.frequencies, 7)
        npt.assert_almost_equal(fap.amplitudes, fap_exp.amplitudes, 7)
        npt.assert_almost_equal(fap.phases, fap_exp.phases, 7)

        for k in ["gain", "units_in", "units_out", "name", "comments"]:
            self.assertEqual(getattr(fap, k), getattr(fap_exp, k))

    def test_coefficient(self):
        self.hx = self.m.get_channel("FL001", "a", "hx", "test")
        coeff = self.hx.channel_response_filter.filters_list[1]
        coeff_exp = self.experiment.surveys[0].filters["v to counts (electric)"]

        self.assertDictEqual(coeff.to_dict(single=True), coeff_exp.to_dict(single=True))

    def tearDown(self):
        self.m.close_mth5()
        self.fn.unlink()
