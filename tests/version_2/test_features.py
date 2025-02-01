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


m = MTH5(file_version="0.2.0")
m.open_mth5(Path(__file__).parent.joinpath("test_feature.h5"))

survey_group = m.add_survey("top_survey")
station_group = m.add_station("mt01", survey="top_survey")
features_group = station_group.features_group
feature = features_group.add_feature_group("coherence")

ts_run = feature.add_feature_run_group("despike", domain="time")
ts_channel = ts_run.add_feature_channel("ex", "electric", None)

fc_run = feature.add_feature_run_group("coherence", domain="frequency")
dl = fc_run.add_decimation_level("0")
