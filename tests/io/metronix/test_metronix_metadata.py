# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:47:59 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5.io.metronix.metronix_metadata import MetronixJSONMagnetic

# =============================================================================


class TestMetronixJSONMagnetic(unittest.TestCase):
    def setUp(self):
        self.fn = Path(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\metronix\mth5_files\small_example\Northern_Mining\stations\Sarıçam\run_001\084_ADU-07e_C002_THx_128Hz.json"
        )
        self.magnetic = MetronixJSONMagnetic(self.fn)

    def test_fn(self):
        self.assertEqual(self.fn, self.magnetic.fn)

    def test_system_number(self):
        self.assertEqual("084", self.magnetic.system_number)

    def test_system_name(self):
        self.assertEqual("ADU-07e", self.magnetic.system_name)

    def test_channel_number(self):
        self.assertEqual(2, self.magnetic.channel_number)

    def test_component(self):
        self.assertEqual("hx", self.magnetic.component)

    def test_sample_rate(self):
        self.assertEqual(128.0, self.magnetic.sample_rate)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
