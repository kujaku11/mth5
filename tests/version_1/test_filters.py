# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:35:30 2021

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

from mth5 import helpers
from mth5.mth5 import MTH5
from mt_metadata.timeseries.filters import PoleZeroFilter, CoefficientFilter

fn_path = Path(__file__).parent
# =============================================================================
helpers.close_open_files()


class TestFilters(unittest.TestCase):
    """
    Test filters to make sure get out what is put in
    """

    @classmethod
    def setUpClass(cls):
        cls.fn = fn_path.joinpath("filter_test.h5")
        cls.m_obj = MTH5(file_version="0.1.0")
        cls.m_obj.open_mth5(cls.fn, "w")
        cls.filter_group = cls.m_obj.filters_group

        cls.zpk = PoleZeroFilter()
        cls.zpk.units_in = "counts"
        cls.zpk.units_out = "mV"
        cls.zpk.name = "zpk_test"
        cls.zpk.poles = np.array([1 + 2j, 0, 1 - 2j])
        cls.zpk.zeros = np.array([10 - 1j, 10 + 1j])

        cls.coefficient = CoefficientFilter()
        cls.coefficient.units_in = "volt"
        cls.coefficient.units_out = "milliVolt per meter"
        cls.coefficient.name = "coefficient_test"
        cls.coefficient.gain = 10.0

        cls.zpk_group = cls.filter_group.add_filter(cls.zpk)
        cls.coefficient_group = cls.filter_group.add_filter(cls.coefficient)

    def test_zpk_in(self):

        self.assertIn("zpk_test", self.filter_group.zpk_group.groups_list)

    def test_zpk_name(self):
        self.assertEqual(self.zpk_group.attrs["name"], self.zpk.name)

    def test_zpk_units_in(self):
        self.assertEqual(self.zpk_group.attrs["units_in"], self.zpk.units_in)

    def test_zpk_units_out(self):
        self.assertEqual(self.zpk_group.attrs["units_out"], self.zpk.units_out)

    def test_zpk_poles(self):
        self.assertTrue(np.allclose(self.zpk_group["poles"][()], self.zpk.poles))

    def test_zpk_zeros(self):
        self.assertTrue(np.allclose(self.zpk_group["zeros"], self.zpk.zeros))

    def test_zpk_out(self):
        new_zpk = self.filter_group.to_filter_object(self.zpk.name)

        self.assertTrue(new_zpk == self.zpk)

    def test_coefficient_in(self):

        self.assertIn(
            "coefficient_test", self.filter_group.coefficient_group.groups_list
        )

    def test_coefficient_name(self):
        self.assertEqual(self.coefficient_group.attrs["name"], self.coefficient.name)

    def test_coefficient_units_in(self):
        self.assertEqual(
            self.coefficient_group.attrs["units_in"], self.coefficient.units_in
        )

    def test_coefficient_units_out(self):
        self.assertEqual(
            self.coefficient_group.attrs["units_out"],
            self.coefficient.units_out,
        )

    def test_coefficient_out(self):
        new_coefficient = self.filter_group.to_filter_object(self.coefficient.name)

        self.assertTrue(new_coefficient == self.coefficient)

    @classmethod
    def tearDownClass(cls):
        cls.m_obj.close_mth5()
        cls.fn.unlink()
