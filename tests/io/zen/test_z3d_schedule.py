# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 21:00:28 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path

from mth5.io.zen import Z3DSchedule

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()), "local files"
)
class TestZ3DHeader(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = r"c:\Users\jpeacock\OneDrive - DOI\mt\example_z3d_data\bm100_20220517_131017_256_EY.Z3D"
        self.z3d_obj = Z3DSchedule(fn=self.fn)
        self.z3d_obj.read_schedule()

    def test_AutoGain(self):
        self.assertEqual(getattr(self.z3d_obj, "AutoGain"), "N")

    def test_Comment(self):
        self.assertEqual(getattr(self.z3d_obj, "Comment"), "")

    def test_Date(self):
        self.assertEqual(getattr(self.z3d_obj, "Date"), "2022-05-17")

    def test_Duty(self):
        self.assertEqual(getattr(self.z3d_obj, "Duty"), "0")

    def test_FFTStacks(self):
        self.assertEqual(getattr(self.z3d_obj, "FFTStacks"), "0")

    def test_Filename(self):
        self.assertEqual(getattr(self.z3d_obj, "Filename"), "")

    def test_Gain(self):
        self.assertEqual(getattr(self.z3d_obj, "Gain"), "1.0000")

    def test_Log(self):
        self.assertEqual(getattr(self.z3d_obj, "Log"), "Y")

    def test_NewFile(self):
        self.assertEqual(getattr(self.z3d_obj, "NewFile"), "Y")

    def test_Period(self):
        self.assertEqual(getattr(self.z3d_obj, "Period"), "0")

    def test_RadioOn(self):
        self.assertEqual(getattr(self.z3d_obj, "RadioOn"), "X")

    def test_SR(self):
        self.assertEqual(getattr(self.z3d_obj, "SR"), "256")

    def test_SamplesPerAcq(self):
        self.assertEqual(getattr(self.z3d_obj, "SamplesPerAcq"), "0")

    def test_Sleep(self):
        self.assertEqual(getattr(self.z3d_obj, "Sleep"), "N")

    def test_Sync(self):
        self.assertEqual(getattr(self.z3d_obj, "Sync"), "Y")

    def test_Time(self):
        self.assertEqual(getattr(self.z3d_obj, "Time"), "13:10:15")

    def test_fn(self):
        self.assertEqual(getattr(self.z3d_obj, "fn"), self.fn)

    def test_initial_start(self):
        self.assertEqual(
            getattr(self.z3d_obj, "initial_start"), "2022-05-17T13:09:57+00:00"
        )

    def test_meta_string(self):
        self.assertEqual(
            getattr(self.z3d_obj, "meta_string"),
            b"\n\n\nGPS Brd339/Brd357 Schedule Details\nSchedule.Date = 2022-05-17\nSchedule.Time = 13:10:15\nSchedule.Sync = Y\nSchedule.NewFile = Y\nSchedule.Period = 0\nSchedule.Duty = 0\nSchedule.S/R = 256\nSchedule.Gain = 1.0000\nSchedule.SamplesPerAcq = 0\nSchedule.FFTStacks = 0\nSchedule.Log = Y\nSchedule.Sleep = N\nSchedule.RadioOn = X\nSchedule.AutoGain = N\nSchedule.Filename = \nSchedule.Comment = \n\n\x00                                                                                                                                \r\n\x00",
        )


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
