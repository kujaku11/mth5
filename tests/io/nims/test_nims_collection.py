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

import pandas as pd
import numpy as np

from mth5.io.nims import NIMSCollection

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()), "local files"
)
class TestNIMSCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nc = NIMSCollection(r"c:\Users\jpeacock\OneDrive - DOI\mt\nims")

        self.df = self.nc.to_dataframe([8])
        self.runs = self.nc.get_runs([8])

        self.station = self.df.station.unique()[0]

    def test_file_path(self):
        self.assertIsInstance(self.nc.file_path, Path)

    def test_get_files(self):
        self.assertListEqual(
            ["mnp300a.BIN", "mnp300b.BIN"],
            [fn.name for fn in self.nc.get_files(self.nc.file_ext)],
        )

    def test_df_columns(self):
        self.assertListEqual(
            self.nc._columns,
            self.df.columns.to_list(),
        )

    def test_df_shape(self):
        self.assertEqual(self.df.shape, (2, 14))

    def test_df_types(self):
        self.df = self.nc._set_df_dtypes(self.df)
        with self.subTest("start"):
            self.assertTrue(
                self.df.start.dtype.type
                == pd._libs.tslibs.timestamps.Timestamp
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
            ["mnp300a", "mnp300b"],
            self.df.run.to_list(),
        )

    def test_runs_keys(self):
        self.assertListEqual(
            list(self.runs[self.station].keys()),
            ["mnp300a", "mnp300b"],
        )

    def test_run_dtype(self):
        self.assertIsInstance(self.runs, OrderedDict)

    def test_run_elements(self):
        for key, rdf in self.runs[self.station].items():
            with self.subTest(key):
                self.assertTrue(
                    (self.df[self.df.run == key].eq(rdf).all(axis=0).all())
                )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
