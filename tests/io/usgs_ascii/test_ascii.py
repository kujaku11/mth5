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
        self.asc.read()
        self.maxDiff = None

    def test_channel(self):
        for ch in self.asc.channel_order:
            ch_obj = getattr(self.asc, ch)

            with self.subTest("size"):
                self.assertEqual(955916, ch_obj.ts.size)

            with self.subTest("start"):
                self.assertEqual("2012-08-21T22:02:27+00:00", ch_obj.start)

            with self.subTest("end"):
                self.assertEqual("2012-08-24T16:25:26+00:00", ch_obj.end)

            with self.subTest("sample rate"):
                self.assertEqual(4, ch_obj.sample_rate)

            with self.subTest("station"):
                self.assertEqual("003", ch_obj.station_metadata.id)

            with self.subTest("run"):
                self.assertEqual("rgr003a", ch_obj.run_metadata.id)

            with self.subTest("type"):
                self.assertIsInstance(ch_obj, ChannelTS)

    def test_to_run(self):
        r = self.asc.to_run_ts()

        with self.subTest("size"):
            self.assertEqual(955916, r.dataset.coords["time"].size)

        with self.subTest("start"):
            self.assertEqual("2012-08-21T22:02:27+00:00", r.start)

        with self.subTest("end"):
            self.assertEqual("2012-08-24T16:25:26+00:00", r.end)

        with self.subTest("sample rate"):
            self.assertEqual(4, r.sample_rate)

        with self.subTest("station"):
            self.assertEqual("003", r.station_metadata.id)

        with self.subTest("run"):
            self.assertEqual("rgr003a", r.run_metadata.id)

        with self.subTest("channels"):
            self.assertListEqual(["hx", "hy", "hz", "ex", "ey"], r.channels)

        with self.subTest("type"):
            self.assertIsInstance(r, RunTS)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
