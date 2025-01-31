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
