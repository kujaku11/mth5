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
import numpy as np

from mth5.io.zen import Z3D

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()), "local files"
)
class TestZ3D(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.z3d = Z3D(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\example_z3d_data\bm100_20220517_131017_256_EY.Z3D"
        )
        self.z3d.read_z3d()

    def test_gps_stamps(self):
        self.assertEqual(9885, self.z3d.gps_stamps.size)

    def test_gps_stamps_seconds(self):
        self.assertEqual(
            self.z3d.gps_stamps.size - 1, self.z3d.end - self.z3d.start
        )

    def test_get_gps_stamp_type(self):
        self.assertEqual(
            np.dtype(
                [
                    ("flag0", "<i4"),
                    ("flag1", "<i4"),
                    ("time", "<i4"),
                    ("lat", "<f8"),
                    ("lon", "<f8"),
                    ("gps_sens", "<i4"),
                    ("num_sat", "<i4"),
                    ("temperature", "<f4"),
                    ("voltage", "<f4"),
                    ("num_fpga", "<i4"),
                    ("num_adc", "<i4"),
                    ("pps_count", "<i4"),
                    ("dac_tune", "<i4"),
                    ("block_len", "<i4"),
                ]
            ),
            self.z3d._gps_dtype,
        )

    def test_gps_stamp_length(self):
        self.assertEqual(self.z3d._gps_stamp_length, 64)

    def test_gps_bytes(self):
        self.assertEqual(self.z3d._gps_bytes, 16)

    def test_gps_flag_0(self):
        self.assertEqual(self.z3d._gps_flag_0, 2147483647)

    def test_gps_flag_1(self):
        self.assertEqual(self.z3d._gps_flag_1, -2147483648)

    def test_block_len(self):
        self.assertEqual(self.z3d._block_len, 65536)

    def test_gps_flag(self):
        self.assertEqual(
            self.z3d.gps_flag, b"\xff\xff\xff\x7f\x00\x00\x00\x80"
        )

    def test_sample_rate(self):
        self.assertEqual(self.z3d.sample_rate, 256)

    def test_get_gps_time(self):
        self.assertTupleEqual(
            self.z3d.get_gps_time(220216, 2210), (215.056, 2210.0)
        )

    def test_get_utc_date_time(self):
        self.assertEqual(
            self.z3d.get_UTC_date_time(2210, 220216),
            "2022-05-17T13:09:58+00:00",
        )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
