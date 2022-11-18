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

from mth5.io.phoenix import PhoenixCollection

# =============================================================================


@unittest.skipIf("peacock" not in str(Path(__file__).as_posix()), "local files")
class TestPhoenixCollection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pc = PhoenixCollection(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\phoenix_example_data\10291_2019-09-06-015630"
        )

        self.df = self.pc.to_dataframe([150, 24000])
        self.runs = self.pc.get_runs([150, 24000])

        self.station = self.df.station.unique()[0]

    def test_file_path(self):
        self.assertIsInstance(self.pc.file_path, Path)

    def test_get_files(self):
        self.assertEqual(
            992,
            len(
                [
                    fn.name
                    for fn in self.pc.get_files(
                        self.pc._file_extension_map[150]
                    )
                ]
            ),
        )

    def test_df_columns(self):
        self.assertListEqual(
            self.pc._columns,
            self.df.columns.to_list(),
        )

    def test_df_shape(self):
        self.assertEqual(self.df.shape, (1984, 14))

    def test_df_types(self):
        self.df = self.pc._set_df_dtypes(self.df)
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
        self.assertTrue(
            (
                self.df.survey
                == list(self.pc.metadata_dict.values())[0].survey_metadata.id
            ).all()
        )

    def test_df_run_names_150(self):
        self.assertEqual(
            "sr150_0001", self.df[self.df.sample_rate == 150].run.unique()[0]
        )

    def test_df_run_names_24k(self):
        run_names = self.df[self.df.sample_rate == 24000].run.unique()

        with self.subTest("len"):
            self.assertEqual(124, run_names.size)

        with self.subTest("first"):
            self.assertEqual(run_names[0], "sr24k_0001")

        with self.subTest("last"):
            self.assertEqual(run_names[-1], "sr24k_0124")

    def test_runs_keys(self):
        self.assertEqual(125, len(self.runs[self.station].keys()))

    def test_run_dtype(self):
        self.assertIsInstance(self.runs, OrderedDict)

    def test_run_elements(self):
        for key, rdf in self.runs[self.station].items():
            with self.subTest(key):
                test_rdf = self.df[self.df.run == key]
                self.assertTrue(
                    (
                        self.df[self.df.run == key]
                        .iloc[0:8]
                        .eq(rdf)
                        .all(axis=0)
                        .all()
                    )
                )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
