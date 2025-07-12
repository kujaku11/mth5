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
    def setUpClass(self):

        self.m = MTH5(file_version="0.2.0")
        self.mth5_fn = Path(__file__).parent.joinpath("test_feature.h5")
        self.m.open_mth5(self.mth5_fn)

        self.survey_group = self.m.add_survey("top_survey")
        self.station_group = self.m.add_station("mt01", survey="top_survey")
        self.features_group = self.station_group.features_group

        # add frequency feature, say coherence
        self.feature_fc = self.features_group.add_feature_group("feature_fc")
        self.fc_run = self.feature_fc.add_feature_run_group(
            "coherence", domain="frequency"
        )
        self.dl = self.fc_run.add_decimation_level("0")
        self.feature_ch = self.dl.add_channel("ex_hy")

        # add time series feature say despiking
        self.feature_ts = self.features_group.add_feature_group("feature_ts")
        self.ts_run = self.feature_ts.add_feature_run_group(
            "despike", domain="time"
        )
        self.ts_channel = self.ts_run.add_feature_channel(
            "ex", "electric", None
        )

    def test_feature_group(self):
        self.assertEqual(
            self.features_group.metadata.mth5_type, "MasterFeatures"
        )

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
    def tearDownClass(self):
        self.m.close_mth5()
        self.mth5_fn.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
