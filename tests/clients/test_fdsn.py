# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:48:24 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5.clients.fdsn import FDSN


# =============================================================================


class TestClientBase(unittest.TestCase):
    def setUp(self):
        self.file_path = Path(__file__)
        self.base = FDSN(self.file_path.parent, **{"h5_mode": "w", "h5_driver": "sec2"})

    def test_h5_kwargs(self):
        keys = [
            "compression",
            "compression_opts",
            "data_level",
            "driver",
            "file_version",
            "fletcher32",
            "mode",
            "shuffle",
        ]
        self.assertListEqual(keys, sorted(self.base.h5_kwargs.keys()))


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
