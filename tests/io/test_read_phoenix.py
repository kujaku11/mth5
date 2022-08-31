# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:39:30 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from pathlib import Path
from mth5.io.phoenix import read_phoenix, PhoenixCollection

# =============================================================================


@unittest.skipIf("peacock" in Path(__file__), "Only local files, cannot test in GitActions")