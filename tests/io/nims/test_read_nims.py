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

from mth5.io.nims import read_nims

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()),
    "Big data on local machine",
)
class TestReadNIMS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nims_obj = read_nims(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\nims\mnp300a.BIN"
        )
