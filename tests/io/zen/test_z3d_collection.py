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


try:
    import mth5_test_data

    zen_path = mth5_test_data.get_test_data_path("zen")
except ImportError:
    raise unittest.SkipTest("mth5_test_data not available")


# =============================================================================


# @unittest.skipIf("peacock" not in str(Path(__file__).as_posix()), "local files")
class TestZ3DCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.zc = Z3DCollection(zen_path)

        cls.df = cls.zc.to_dataframe([256, 4096, 1024])
        cls.runs = cls.zc.get_runs([256, 4096, 1024])
        cls.station = cls.df.station.unique()[0]
        cls.maxDiff = None

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
        df = self.zc._set_df_dtypes(self.df)
        with self.subTest("start"):
            self.assertTrue(df.start.dtype == pd.api.types.is_datetime64_any_dtype)
        with self.subTest("end"):
            self.assertTrue(df.end.dtype == pd.api.types.is_datetime64_any_dtype)

        with self.subTest("instrument_id"):
            self.assertTrue(df.instrument_id.dtype.type == np.object_)
        with self.subTest("calibration_fn"):
            self.assertTrue(df.calibration_fn.dtype.type == np.object_)

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
