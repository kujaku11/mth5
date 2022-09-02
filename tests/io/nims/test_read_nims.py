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


from mth5.io.nims import NIMS, read_nims
from mt_metadata.utils.mttime import MTime

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()),
    "Big data on local machine",
)
class TestReadNIMS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.nims_obj = NIMS(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\nims\mnp300a.BIN"
        )
        self.nims_obj.read_nims()
        self.maxDiff = None

    def test_site_name(self):
        self.assertEqual(self.nims_obj.site_name, "Budwieser Spring")

    def test_state_province(self):
        self.assertEqual(self.nims_obj.state_province, "CA")

    def test_country(self):
        self.assertEqual(self.nims_obj.country, "USA")

    def test_box_id(self):
        self.assertEqual(self.nims_obj.box_id, "1105-3")

    def test_mag_id(self):
        self.assertEqual(self.nims_obj.mag_id, "1305-3")

    def test_ex_length(self):
        self.assertEqual(self.nims_obj.ex_length, 109.0)

    def test_ex_azimuth(self):
        self.assertEqual(self.nims_obj.ex_azimuth, 0.0)

    def test_ey_length(self):
        self.assertEqual(self.nims_obj.ey_length, 101.0)

    def test_ey_azimuth(self):
        self.assertEqual(self.nims_obj.ey_azimuth, 90.0)

    def test_n_electrode_id(self):
        self.assertEqual(self.nims_obj.n_electrode_id, "1")

    def test_s_electrode_id(self):
        self.assertEqual(self.nims_obj.s_electrode_id, "2")

    def test_e_electrode_id(self):
        self.assertEqual(self.nims_obj.e_electrode_id, "3")

    def test_w_electrode_id(self):
        self.assertEqual(self.nims_obj.w_electrode_id, "4")

    def test_ground_electrode_info(self):
        self.assertEqual(self.nims_obj.ground_electrode_info, None)

    def test_header_gps_stamp(self):
        self.assertEqual(
            self.nims_obj.header_gps_stamp, MTime("2019-09-26 18:29:29")
        )

    def test_header_gps_latitude(self):
        self.assertEqual(self.nims_obj.header_gps_latitude, 34.7268)

    def test_header_gps_longitude(self):
        self.assertEqual(self.nims_obj.header_gps_longitude, -115.735)

    def test_header_gps_elevation(self):
        self.assertEqual(self.nims_obj.header_gps_elevation, 939.8)

    def test_operator(self):
        self.assertEqual(self.nims_obj.operator, "KP")

    def test_comments(self):
        self.assertEqual(
            self.nims_obj.comments,
            (
                "N/S: CRs: .764/.769 DCV: 3.5 ACV:1 \n"
                "E/W:  CRs: .930/.780 DCV: 28.2 ACV: 1\n"
                "Hwy 40 800m S, X array"
            ),
        )

    def test_run_id(self):
        self.assertEqual(self.nims_obj.run_id, "Mnp300a")

    def test_data_start_seek(self):
        self.assertEqual(self.nims_obj.data_start_seek, 946)

    def test_block_size(self):
        self.assertEqual(self.nims_obj.block_size, 131)

    def test_block_sequence(self):
        self.assertEqual(self.nims_obj.block_sequence, [1, 131])

    def test_sample_rate(self):
        self.assertEqual(self.nims_obj.sample_rate, 8)

    def test_e_conversion_factor(self):
        self.assertEqual(
            self.nims_obj.e_conversion_factor, 2.44141221047903e-06
        )

    def test_h_conversion_factor(self):
        self.assertEqual(self.nims_obj.h_conversion_factor, 0.01)

    def test_t_conversion_factor(self):
        self.assertEqual(self.nims_obj.t_conversion_factor, 70)

    def test_t_offset(self):
        self.assertEqual(self.nims_obj.t_offset, 18048)

    def test_ground_electrodeinfo(self):
        self.assertEqual(self.nims_obj.ground_electrodeinfo, "Cu")


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
