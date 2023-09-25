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

from mth5.io.lemi import LEMICollection

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()), "local files"
)
class TestLEMICollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.lc = LEMICollection(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\lemi\DATA0110"
        )
        self.lc.station_id = "mt01"
        self.lc.survey_id = "test"

        self.df = self.lc.to_dataframe()
        self.df = self.df.fillna(0)
        self.runs = self.lc.get_runs([1])
        self.maxDiff = None

    def test_file_path(self):
        self.assertIsInstance(self.lc.file_path, Path)

    def test_get_files(self):
        self.assertListEqual(
            [
                "202009302021.TXT",
                "202009302029.TXT",
                "202009302054.TXT",
                "202009302112.TXT",
                "202009302114.TXT",
                "202010010000.TXT",
                "202010020000.TXT",
                "202010030000.TXT",
                "202010040000.TXT",
                "202010050000.TXT",
                "202010060000.TXT",
                "202010070000.TXT",
            ],
            [fn.name for fn in self.lc.get_files(self.lc.file_ext)],
        )

    def test_df_columns(self):
        self.assertListEqual(
            self.lc._columns,
            self.df.columns.to_list(),
        )

    def test_df_shape(self):
        self.assertEqual(self.df.shape, (12, 19))

    def test_df_types(self):
        self.df = self.lc._set_df_dtypes(self.df)
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

    def test_survey_id(self):
        self.assertTrue((self.df.survey == self.lc.survey_id).all())

    def test_df_run_names(self):
        self.assertListEqual(
            [
                "sr1_0001",
                "sr1_0002",
                "sr1_0003",
                "sr1_0004",
                "sr1_0005",
                "sr1_0005",
                "sr1_0005",
                "sr1_0005",
                "sr1_0005",
                "sr1_0005",
                "sr1_0005",
                "sr1_0005",
            ],
            self.df.run.to_list(),
        )

    def test_runs_keys(self):
        self.assertListEqual(
            list(self.runs[self.lc.station_id].keys()),
            ["sr1_0001", "sr1_0002", "sr1_0003", "sr1_0004", "sr1_0005"],
        )

    def test_run_dtype(self):
        self.assertIsInstance(self.runs, OrderedDict)

    def test_run_elements(self):
        for key, rdf in self.runs[self.lc.station_id].items():
            rdf = rdf.fillna(0)
            with self.subTest(key):
                self.assertTrue(
                    (self.df[self.df.run == key].eq(rdf).all(axis=0).all())
                )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
