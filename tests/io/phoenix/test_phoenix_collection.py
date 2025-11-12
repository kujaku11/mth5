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

from mth5.io.phoenix import PhoenixCollection


try:
    import mth5_test_data

    phx_data_path = mth5_test_data.get_test_data_path("phoenix") / "sample_data"
    has_test_data = True
except ImportError:
    has_test_data = False
# =============================================================================


# @unittest.skipIf("peacock" not in str(Path(__file__).as_posix()), "local files")
class TestPhoenixCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pc = PhoenixCollection(phx_data_path / "10128_2021-04-27-032436")

        cls.df = cls.pc.to_dataframe([150, 24000])
        cls.df = cls.df.fillna(0)
        cls.runs = cls.pc.get_runs([150, 24000])

        cls.station = cls.df.station.unique()[0]

    def test_file_path(self):
        self.assertIsInstance(self.pc.file_path, Path)

    def test_get_files(self):
        self.assertEqual(
            8,
            len(
                [fn.name for fn in self.pc.get_files(self.pc._file_extension_map[150])]
            ),
        )

    def test_df_columns(self):
        self.assertListEqual(
            self.pc._columns,
            self.df.columns.to_list(),
        )

    def test_df_shape(self):
        self.assertEqual(self.df.shape, (12, 19))

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
            self.assertEqual(1, run_names.size)

        with self.subTest("first"):
            self.assertEqual(run_names[0], "sr24k_0001")

        with self.subTest("last"):
            self.assertEqual(run_names[-1], "sr24k_0001")

    def test_runs_keys(self):
        self.assertEqual(2, len(self.runs[self.station].keys()))

    def test_run_dtype(self):
        self.assertIsInstance(self.runs, OrderedDict)

    def test_run_elements(self):
        for key, rdf in self.runs[self.station].items():
            rdf = rdf.fillna(0)
            with self.subTest(key):
                self.assertTrue(
                    (self.df[self.df.run == key].iloc[0:4].eq(rdf).all(axis=0).all())
                )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
