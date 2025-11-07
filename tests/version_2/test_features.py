# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:31:22 2025

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import unittest
from mth5.mth5 import MTH5

# =============================================================================


class TestFeaturesSimple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.m = MTH5(file_version="0.2.0")
        cls.mth5_fn = Path(__file__).parent.joinpath("test_feature.h5")
        cls.m.open_mth5(cls.mth5_fn)

        cls.survey_group = cls.m.add_survey("top_survey")
        cls.station_group = cls.m.add_station("mt01", survey="top_survey")
        cls.features_group = cls.station_group.features_group

        # add frequency feature, say coherence
        cls.feature_fc = cls.features_group.add_feature_group("feature_fc")
        cls.fc_run = cls.feature_fc.add_feature_run_group(
            "coherence", domain="frequency"
        )
        cls.dl = cls.fc_run.add_decimation_level("0")
        cls.feature_ch = cls.dl.add_channel("ex_hy")

        # add time series feature say despiking
        cls.feature_ts = cls.features_group.add_feature_group("feature_ts")
        cls.ts_run = cls.feature_ts.add_feature_run_group("despike", domain="time")
        cls.ts_channel = cls.ts_run.add_feature_channel("ex", "electric", None)

    def test_feature_group(self):
        self.assertEqual(self.features_group.metadata.mth5_type, "MasterFeatures")

    def test_feature_fc(self):
        self.assertEqual(self.feature_fc.metadata.name, "feature_fc")

    def test_feature_fc_run(self):
        with self.subTest("id"):
            self.assertEqual(self.fc_run.metadata.id, "coherence")

    def test_feature_fc_decimation_level(self):
        self.assertEqual(self.dl.metadata.id, "0")

    def test_feature_fc_decimation_channel(self):
        self.assertEqual(self.feature_ch.metadata.name, "ex_hy")

    def test_feature_ts(self):
        self.assertEqual(self.feature_ts.metadata.name, "feature_ts")

    def test_feature_ts_run(self):
        self.assertEqual(self.ts_run.metadata.id, "despike")

    def test_feature_ts_channel(self):
        self.assertEqual(self.ts_channel.metadata.component, "ex")

    @classmethod
    def tearDownClass(cls):
        cls.m.close_mth5()
        cls.mth5_fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
