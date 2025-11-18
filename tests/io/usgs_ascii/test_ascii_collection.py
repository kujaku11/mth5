# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 18:19:12 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
import warnings
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from mth5.io.usgs_ascii import USGSasciiCollection


try:
    import mth5_test_data

    ascii_data_path = mth5_test_data.get_test_data_path("usgs_ascii")
except ImportError:
    ascii_data_path = None


# =============================================================================


@unittest.skipIf(ascii_data_path is None, "local files")
class TestUSGSasciiCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nc = USGSasciiCollection(ascii_data_path)
        self.df = self.nc.to_dataframe([4])
        # Suppress FutureWarnings for this deprecated behavior while using proper approach
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.df = self.df.fillna(0)
        self.runs = self.nc.get_runs([4])
        self.station = self.df.station.unique()[0]

    def test_file_path(self):
        self.assertIsInstance(self.nc.file_path, Path)

    def test_get_files(self):
        self.assertListEqual(
            ["rgr006a_converted.asc"],
            [fn.name for fn in self.nc.get_files(self.nc.file_ext)],
        )

    def test_df_columns(self):
        self.assertListEqual(
            self.nc._columns,
            self.df.columns.to_list(),
        )

    def test_df_shape(self):
        self.assertEqual(self.df.shape, (1, 19))

    def test_df_types(self):
        # Create a copy for testing to avoid modifying the shared self.df
        test_df = self.nc._set_df_dtypes(self.df.copy())
        with self.subTest("start"):
            # More robust way to test for datetime dtype
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(test_df.start))
            # Also test that it's not null/empty
            self.assertFalse(test_df.start.isna().all())

        with self.subTest("end"):
            # More robust way to test for datetime dtype
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(test_df.end))
            # Also test that it's not null/empty
            self.assertFalse(test_df.end.isna().all())

        with self.subTest("instrument_id"):
            # More robust way to test for object/string dtype
            self.assertTrue(pd.api.types.is_object_dtype(test_df.instrument_id))
            # Alternative: can also test for string specifically if needed
            # self.assertTrue(pd.api.types.is_string_dtype(test_df.instrument_id))

        with self.subTest("calibration_fn"):
            # More robust way to test for object/string dtype
            self.assertTrue(pd.api.types.is_object_dtype(test_df.calibration_fn))
            # Alternative: can also test for string specifically if needed
            # self.assertTrue(pd.api.types.is_string_dtype(test_df.calibration_fn))

    def test_df_run_names(self):
        self.assertListEqual(
            ["rgr006a"],
            self.df.run.to_list(),
        )

    def test_runs_keys(self):
        self.assertListEqual(
            list(self.runs[self.station].keys()),
            ["rgr006a"],
        )

    def test_run_dtype(self):
        self.assertIsInstance(self.runs, OrderedDict)

    def test_run_elements(self):
        for key, rdf in self.runs[self.station].items():
            # Suppress FutureWarnings for this deprecated behavior while using proper approach
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                rdf = rdf.fillna(0)
            with self.subTest(key):
                self.assertTrue((self.df[self.df.run == key].eq(rdf).all(axis=0).all()))


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
