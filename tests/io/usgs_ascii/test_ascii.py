# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:38:10 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5.io.usgs_ascii import USGSasc
from mth5.timeseries import ChannelTS, RunTS

# =============================================================================


@unittest.skipIf("peacock" not in str(Path(__file__).as_posix()), "local file")
class TestUSGSAscii(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.asc = USGSasc(
            fn=r"C:\Users\jpeacock\OneDrive - DOI\mt\usgs_ascii\rgr003a_converted.asc"
        )
        self.asc.read_ascii_file()
        self.maxDiff = None

    def test_ex(self):
        pass


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
