# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 18:19:12 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from mth5.io.zen import Z3DCollection


# =============================================================================


@unittest.skipIf("peacock" not in str(Path(__file__).as_posix()), "local files")
class TestZ3DCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.zc = Z3DCollection(r"c:\Users\jpeacock\OneDrive - DOI\mt\example_z3d_data")

        self.df = self.zc.to_dataframe([256, 4096])
        self.runs = self.zc.get_runs([256, 4096])

        self.station = self.df.station.unique()[0]
        self.maxDiff = None

    def test_file_path(self):
        self.assertIsInstance(self.zc.file_path, Path)

    def test_get_files(self):
        self.assertEqual(10, len(self.zc.get_files(self.zc.file_ext)))

    def test_df_columns(self):
        self.assertListEqual(
            self.zc._columns,
            self.df.columns.to_list(),
        )

    def test_df_shape(self):
        self.assertEqual(self.df.shape, (10, 19))

    def test_df_types(self):
        self.df = self.zc._set_df_dtypes(self.df)
        with self.subTest("start"):
            self.assertTrue(
                self.df.start.dtype.type == pd._libs.tslibs.timestamps.Timestamp
            )
        with self.subTest("end"):
            self.assertTrue(
                self.df.end.dtype.type == pd._libs.tslibs.timestamps.Timestamp
            )

        with self.subTest("instrument_id"):
            self.assertTrue(self.df.instrument_id.dtype.type == np.object_)

        with self.subTest("calibration_fn"):
            self.assertTrue(self.df.calibration_fn.dtype.type == np.object_)

    def test_survey_id(self):
        self.assertTrue((self.df.survey == "").all())

    def test_df_run_names_256(self):
        self.assertEqual(
            "sr256_0002", self.df[self.df.sample_rate == 256].run.unique()[0]
        )

    def test_df_run_names_4096(self):
        self.assertEqual(
            "sr4096_0001", self.df[self.df.sample_rate == 4096].run.unique()[0]
        )

    def test_run_dtype(self):
        self.assertIsInstance(self.runs, OrderedDict)

    def test_run_elements(self):
        for key, rdf in self.runs[self.station].items():
            with self.subTest(key):
                test_rdf = self.df[self.df.run == key]
                rdf = rdf.fillna(0)
                test_rdf = test_rdf.fillna(0)
                self.assertTrue((test_rdf.iloc[0:8].eq(rdf).all(axis=0).all()))


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
