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
from mth5.mth5 import MTH5

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mt_metadata import STATIONXML_FAP

fn_path = Path(__file__).parent


class TestFAPMTH5(unittest.TestCase):
    """
    Test making an MTH5 file from a FAP filtered StationXML

    """

    @classmethod
    def setUpClass(cls):
        cls.translator = XMLInventoryMTExperiment()
        cls.experiment = cls.translator.xml_to_mt(stationxml_fn=STATIONXML_FAP)

        cls.fn = fn_path.joinpath("from_fap_stationxml.h5")
        # if cls.fn.exists():
        #     cls.fn.unlink()

        cls.m = MTH5(file_version="0.1.0")
        cls.m.open_mth5(cls.fn, mode="a")
        cls.m.from_experiment(cls.experiment, 0)

        cls.initial_has_entries = cls.m.channel_summary._has_entries()

    def test_has_survey(self):
        self.assertEqual(self.m.has_group("Survey"), True)

    def test_has_station(self):
        self.assertEqual(self.m.has_group("Survey/Stations"), True)
        self.assertEqual(self.m.has_group("Survey/Stations/FL001"), True)

    def test_has_run_a(self):
        self.assertEqual(self.m.has_group("Survey/Stations/FL001/a"), True)

    def test_run_a_has_data(self):
        run_a = self.m.get_run("FL001", "a")
        self.assertEqual(run_a.has_data(), False)

    def test_has_run_b(self):
        self.assertEqual(self.m.has_group("Survey/Stations/FL001/b"), True)

    def test_has_hx_a(self):
        self.assertEqual(self.m.has_group("Survey/Stations/FL001/a/hx"), True)

    def test_has_hx_b(self):
        self.assertEqual(self.m.has_group("Survey/Stations/FL001/b/hx"), True)

    def test_has_fap_table(self):

        self.assertEqual(
            self.m.has_group("Survey/Filters/fap/frequency response table_00"),
            True,
        )

    def test_has_coefficient_filter(self):
        self.assertEqual(
            self.m.has_group("Survey/Filters/coefficient/v to counts (electric)"),
            True,
        )

    def test_get_channel(self):
        self.hx = self.m.get_channel("FL001", "a", "hx")
        fnames = [f.name for f in self.hx.channel_response.filters_list]

        with self.subTest("fap filter name"):
            self.assertIn("frequency response table_00", fnames)
        with self.subTest("counts filter name"):
            self.assertIn("v to counts (electric)", fnames)
        with self.subTest("channel has data"):
            self.assertEqual(self.hx.has_data(), False)

    def test_fap(self):
        self.hx = self.m.get_channel("FL001", "a", "hx")
        fap = self.hx.channel_response.filters_list[0]
        fap_exp = self.experiment.surveys[0].filters["frequency response table_00"]

        with self.subTest("frequencies"):
            self.assertTrue(np.allclose(fap.frequencies, fap_exp.frequencies, 7))
        with self.subTest("amplitude"):
            self.assertTrue(np.allclose(fap.amplitudes, fap_exp.amplitudes, 7))
        with self.subTest("phase"):
            self.assertTrue(np.allclose(fap.phases, fap_exp.phases, 7))

        with self.subTest("np frequencies"):
            npt.assert_almost_equal(fap.frequencies, fap_exp.frequencies, 7)
        with self.subTest("np amplitude"):
            npt.assert_almost_equal(fap.amplitudes, fap_exp.amplitudes, 7)
        with self.subTest("np phase"):
            npt.assert_almost_equal(fap.phases, fap_exp.phases, 7)

        for k in ["gain", "units_in", "units_out", "name", "comments"]:
            with self.subTest(k):
                self.assertEqual(getattr(fap, k), getattr(fap_exp, k))

    def test_coefficient(self):
        self.hx = self.m.get_channel("FL001", "a", "hx")
        coeff = self.hx.channel_response.filters_list[1]
        coeff_exp = self.experiment.surveys[0].filters["v to counts (electric)"]

        self.assertDictEqual(coeff.to_dict(single=True), coeff_exp.to_dict(single=True))

    def test_has_entries(self):
        self.assertEqual(False, self.initial_has_entries)

    def test_run_summary_has_data(self):
        run_summary = self.m.run_summary
        self.assertListEqual(run_summary.has_data.values.tolist(), [False, False])

    @classmethod
    def tearDownClass(cls):
        cls.m.close_mth5()
        cls.fn.unlink()


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
