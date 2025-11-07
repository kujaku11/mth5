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

from mth5.io.usgs_ascii import USGSasciiCollection


# =============================================================================


@unittest.skipIf("peacock" not in str(Path(__file__).as_posix()), "local files")
class TestUSGSasciiCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nc = USGSasciiCollection(r"c:\Users\jpeacock\OneDrive - DOI\mt\usgs_ascii")

        self.df = self.nc.to_dataframe([4])
        self.df = self.df.fillna(0)
        self.runs = self.nc.get_runs([4])
        self.station = self.df.station.unique()[0]

    def test_file_path(self):
        self.assertIsInstance(self.nc.file_path, Path)

    def test_get_files(self):
        self.assertListEqual(
            ["rgr003a_converted.asc", "rgr003b_converted.asc"],
            [fn.name for fn in self.nc.get_files(self.nc.file_ext)],
        )

    def test_df_columns(self):
        self.assertListEqual(
            self.nc._columns,
            self.df.columns.to_list(),
        )

    def test_df_shape(self):
        self.assertEqual(self.df.shape, (2, 19))

    def test_df_types(self):
        self.df = self.nc._set_df_dtypes(self.df)
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

    def test_df_run_names(self):
        self.assertListEqual(
            ["rgr003a", "rgr003b"],
            self.df.run.to_list(),
        )

    def test_runs_keys(self):
        self.assertListEqual(
            list(self.runs[self.station].keys()),
            ["rgr003a", "rgr003b"],
        )

    def test_run_dtype(self):
        self.assertIsInstance(self.runs, OrderedDict)

    def test_run_elements(self):
        for key, rdf in self.runs[self.station].items():
            rdf = rdf.fillna(0)
            with self.subTest(key):
                self.assertTrue((self.df[self.df.run == key].eq(rdf).all(axis=0).all()))


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
