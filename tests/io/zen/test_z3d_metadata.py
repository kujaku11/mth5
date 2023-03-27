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

from mth5.io.zen import Z3DMetadata

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()), "local files"
)
class TestZ3DMetadataEy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = r"c:\Users\jpeacock\OneDrive - DOI\mt\example_z3d_data\bm100_20220517_131017_256_EY.Z3D"
        self.z3d_obj = Z3DMetadata(fn=self.fn)
        self.z3d_obj.read_metadata()

    def test_board_cal(self):
        self.assertEqual(getattr(self.z3d_obj, "board_cal"), [])

    def test_cal_ant(self):
        self.assertEqual(getattr(self.z3d_obj, "cal_ant"), None)

    def test_cal_board(self):
        self.assertEqual(
            getattr(self.z3d_obj, "cal_board"),
            {
                "cal.date": "11/19/2021",
                "cal.attn": 0.0,
                256: {
                    "frequency": 2.0,
                    "amplitude": 1.00153,
                    "phase": -1533.33,
                },
                1024: {
                    "frequency": 8.0,
                    "amplitude": 1.001513,
                    "phase": -1532.9,
                },
                4096: {
                    "frequency": 32.0,
                    "amplitude": 1.001414,
                    "phase": -1534.24,
                },
                "gdp.volt": 13.25,
                "gdp.temp": 41.5,
            },
        )

    def test_cal_ver(self):
        self.assertEqual(getattr(self.z3d_obj, "cal_ver"), None)

    def test_ch_azimuth(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_azimuth"), None)

    def test_ch_cmp(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_cmp"), "ey")

    def test_ch_cres(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_cres"), None)

    def test_ch_length(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_length"), None)

    def test_ch_number(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_number"), None)

    def test_ch_offset_xyz1(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_offset_xyz1"), "0:0:0")

    def test_ch_offset_xyz2(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_offset_xyz2"), "0:-56:0")

    def test_ch_sp(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_sp"), "0.02")

    def test_ch_stn(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_stn"), "100")

    def test_ch_vmax(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_vmax"), "2e-05")

    def test_ch_xyz1(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_xyz1"), None)

    def test_ch_xyz2(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_xyz2"), None)

    def test_coil_cal(self):
        self.assertEqual(getattr(self.z3d_obj, "coil_cal"), [])

    def test_count(self):
        self.assertEqual(getattr(self.z3d_obj, "count"), 3)

    def test_find_metadata(self):
        self.assertEqual(getattr(self.z3d_obj, "find_metadata"), False)

    def test_fn(self):
        self.assertEqual(getattr(self.z3d_obj, "fn"), self.fn)

    def test_gdp_number(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_number"), "24")

    def test_gdp_operator(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_operator"), "")

    def test_gdp_progver(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_progver"), "zenacqv4.65")

    def test_gdp_temp(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_temp"), None)

    def test_gdp_volt(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_volt"), None)

    def test_job_by(self):
        self.assertEqual(getattr(self.z3d_obj, "job_by"), "")

    def test_job_for(self):
        self.assertEqual(getattr(self.z3d_obj, "job_for"), "")

    def test_job_name(self):
        self.assertEqual(getattr(self.z3d_obj, "job_name"), "")

    def test_job_number(self):
        self.assertEqual(getattr(self.z3d_obj, "job_number"), "")

    def test_line_name(self):
        self.assertEqual(getattr(self.z3d_obj, "line_name"), None)

    def test_rx_aspace(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_aspace"), None)

    def test_rx_sspace(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_sspace"), None)

    def test_rx_stn(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_stn"), "100")

    def test_rx_xazimuth(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_xazimuth"), "0")

    def test_rx_xyz0(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_xyz0"), None)

    def test_rx_yazimuth(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_yazimuth"), None)

    def test_rx_zpositive(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_zpositive"), "down")

    def test_station(self):
        self.assertEqual(getattr(self.z3d_obj, "station"), "100")

    def test_survey_type(self):
        self.assertEqual(getattr(self.z3d_obj, "survey_type"), None)

    def test_unit_length(self):
        self.assertEqual(getattr(self.z3d_obj, "unit_length"), None)


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()), "local files"
)
class TestZ3DMetadataHx(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = r"c:\Users\jpeacock\OneDrive - DOI\mt\example_z3d_data\bm100_20220517_131017_256_HX.Z3D"
        self.z3d_obj = Z3DMetadata(fn=self.fn)
        self.z3d_obj.read_metadata()

    def test_board_cal(self):
        self.assertEqual(getattr(self.z3d_obj, "board_cal"), [])

    def test_cal_ant(self):
        self.assertEqual(getattr(self.z3d_obj, "cal_ant"), "2314")

    def test_cal_board(self):
        self.assertEqual(
            getattr(self.z3d_obj, "cal_board"),
            {
                "cal.date": "11/19/2021",
                "cal.attn": 0.0,
                256: {
                    "frequency": 2.0,
                    "amplitude": 1.000128,
                    "phase": -1533.34,
                },
                1024: {
                    "frequency": 8.0,
                    "amplitude": 1.000109,
                    "phase": -1532.93,
                },
                4096: {
                    "frequency": 32.0,
                    "amplitude": 1.000011,
                    "phase": -1534.34,
                },
                "gdp.volt": 13.25,
                "gdp.temp": 41.5,
            },
        )

    def test_cal_ver(self):
        self.assertEqual(getattr(self.z3d_obj, "cal_ver"), None)

    def test_ch_antsn(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_antsn"), "2314")

    def test_ch_azimuth(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_azimuth"), "0")

    def test_ch_cmp(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_cmp"), "hx")

    def test_ch_cres(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_cres"), None)

    def test_ch_incl(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_incl"), "0")

    def test_ch_length(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_length"), None)

    def test_ch_number(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_number"), None)

    def test_ch_offset_xyz1(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_offset_xyz1"), "0:0:0")

    def test_ch_sp(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_sp"), "0.02")

    def test_ch_stn(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_stn"), "100")

    def test_ch_vmax(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_vmax"), "2e-05")

    def test_ch_xyz1(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_xyz1"), None)

    def test_ch_xyz2(self):
        self.assertEqual(getattr(self.z3d_obj, "ch_xyz2"), None)

    def test_coil_cal(self):
        self.assertTrue(
            (
                getattr(self.z3d_obj, "coil_cal")
                == np.core.records.fromrecords(
                    np.array(
                        [
                            (7.32422e-04, 0.28228, 1570.2),
                            (9.76563e-04, 0.40325, 1563.3),
                            (1.46484e-03, 0.60488, 1555.5),
                            (1.95313e-03, 0.76618, 1551.0),
                            (2.92969e-03, 1.16944, 1542.8),
                            (3.90625e-03, 1.57269, 1537.1),
                            (5.85938e-03, 2.3792, 1529.2),
                            (7.81250e-03, 3.14538, 1523.8),
                            (1.17188e-02, 4.71807, 1516.0),
                            (1.56250e-02, 6.29076, 1510.5),
                            (2.34375e-02, 9.33806, 1473.0),
                            (3.12500e-02, 12.4627, 1445.4),
                            (4.68750e-02, 18.6755, 1390.9),
                            (6.25000e-02, 24.6371, 1312.7),
                            (9.37500e-02, 35.4436, 1205.7),
                            (1.25000e-01, 44.9365, 1103.1),
                            (1.87500e-01, 59.9935, 923.5),
                            (2.50000e-01, 70.6949, 781.2),
                            (3.75000e-01, 83.106, 583.3),
                            (5.00000e-01, 89.3232, 459.3),
                            (7.50000e-01, 94.7067, 318.1),
                            (1.00000e00, 96.7628, 241.1),
                            (1.50000e00, 98.412, 160.7),
                            (2.00000e00, 98.9519, 119.4),
                            (3.00000e00, 99.3774, 76.6),
                            (4.00000e00, 99.5137, 53.9),
                            (6.00000e00, 99.6679, 29.3),
                            (8.00000e00, 99.6429, 14.9),
                            (1.20000e01, 99.7832, -3.2),
                            (1.60000e01, 99.7335, -16.8),
                            (2.40000e01, 99.7926, -38.3),
                            (3.20000e01, 99.7775, -56.9),
                            (4.80000e01, 99.836, -92.2),
                            (6.40000e01, 99.7502, -126.2),
                            (9.60000e01, 99.8861, -192.3),
                            (1.28000e02, 99.8326, -258.8),
                            (1.92000e02, 99.8494, -392.1),
                            (2.56000e02, 99.8944, -525.6),
                            (3.84000e02, 99.7758, -803.6),
                            (5.12000e02, 99.1085, -1099.6),
                            (7.68000e02, 90.8383, -1767.5),
                            (1.02400e03, 67.8248, -2460.5),
                            (1.53600e03, 26.1416, 2930.7),
                            (2.04800e03, 11.2864, 2549.5),
                            (3.07200e03, 3.50231, 2169.9),
                            (4.09600e03, 1.52457, 1951.5),
                            (6.14400e03, 0.48749, 1748.3),
                            (8.19200e03, 0.22572, 1577.5),
                        ]
                    ),
                    names="frequency, amplitude, phase",
                )
            ).all()
        )

    def test_count(self):
        self.assertEqual(getattr(self.z3d_obj, "count"), 9)

    def test_find_metadata(self):
        self.assertEqual(getattr(self.z3d_obj, "find_metadata"), False)

    def test_fn(self):
        self.assertEqual(getattr(self.z3d_obj, "fn"), self.fn)

    def test_gdp_number(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_number"), "24")

    def test_gdp_operator(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_operator"), "")

    def test_gdp_progver(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_progver"), "zenacqv4.65")

    def test_gdp_temp(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_temp"), None)

    def test_gdp_volt(self):
        self.assertEqual(getattr(self.z3d_obj, "gdp_volt"), None)

    def test_job_by(self):
        self.assertEqual(getattr(self.z3d_obj, "job_by"), "")

    def test_job_for(self):
        self.assertEqual(getattr(self.z3d_obj, "job_for"), "")

    def test_job_name(self):
        self.assertEqual(getattr(self.z3d_obj, "job_name"), "")

    def test_job_number(self):
        self.assertEqual(getattr(self.z3d_obj, "job_number"), "")

    def test_line_name(self):
        self.assertEqual(getattr(self.z3d_obj, "line_name"), None)

    def test_rx_aspace(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_aspace"), None)

    def test_rx_sspace(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_sspace"), None)

    def test_rx_stn(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_stn"), "100")

    def test_rx_xazimuth(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_xazimuth"), "0")

    def test_rx_xyz0(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_xyz0"), None)

    def test_rx_yazimuth(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_yazimuth"), None)

    def test_rx_zpositive(self):
        self.assertEqual(getattr(self.z3d_obj, "rx_zpositive"), "down")

    def test_station(self):
        self.assertEqual(getattr(self.z3d_obj, "station"), "100")

    def test_survey_type(self):
        self.assertEqual(getattr(self.z3d_obj, "survey_type"), None)

    def test_unit_length(self):
        self.assertEqual(getattr(self.z3d_obj, "unit_length"), None)


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
