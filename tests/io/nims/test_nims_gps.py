# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:56:45 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest

from mth5.io.nims import GPS, GPSError

# =============================================================================

gprmc_string = (
    "GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E"
)
gpgga_string = (
    "GPGGA,183511,3443.6098,N,11544.1007,W,1,04,2.6,937.2,M,-28.1,M,,"
)


class TestGPS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gprmc_string = "GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*"
        self.gpgga_string = (
            "GPGGA,183511,3443.6098,N,11544.1007,W,1,04,2.6,937.2,M,-28.1,M,*"
        )
        self.gprmc_bytes = self.gprmc_string.encode()
        self.gpgga_bytes = self.gpgga_string.encode()

        self.gps_obj = GPS("")

        self.a_list = [
            self.gprmc_string,
            self.gpgga_string,
            self.gprmc_string,
            self.gpgga_string,
        ]

        self.b_list = [
            self.gprmc_string,
            self.gpgga_string,
            self.gprmc_bytes,
            self.gpgga_bytes,
        ]

    def test_validate_string(self):

        for aa, bb in zip(self.a_list, self.b_list):

            with self.subTest(aa[0:5]):
                self.assertEqual(
                    aa[:-1],
                    self.gps_obj.validate_gps_string(bb),
                )

    def test_split_list(self):
        for aa, bb in zip(self.a_list, self.b_list):

            with self.subTest(aa[0:5]):
                self.assertListEqual(
                    aa[:-1].split(","),
                    self.gps_obj._split_gps_string(bb),
                )

    def test_validate_list(self):
        for aa, bb in zip(self.a_list, self.b_list):
            input_list = self.gps_obj._split_gps_string(bb)
            gps_list, error_list = self.gps_obj.validate_gps_list(input_list)
            with self.subTest(aa[0:5]):

                self.assertListEqual(aa[:-1].split(","), gps_list)

            with self.subTest("error list"):
                self.assertListEqual([], error_list)

    def test_validate_latitude_fail(self):
        with self.subTest("length"):
            self.assertRaises(
                GPSError, self.gps_obj._validate_latitude, "3443.60", "N"
            )

        with self.subTest("hemisphere length"):
            self.assertRaises(
                GPSError, self.gps_obj._validate_latitude, "3443.6098", "NS"
            )

        with self.subTest("hemisphere"):
            self.assertRaises(
                GPSError, self.gps_obj._validate_latitude, "3443.6098", "Z"
            )

    def test_validate_longitude_fail(self):
        with self.subTest("length"):
            self.assertRaises(
                GPSError, self.gps_obj._validate_longitude, "115.17", "W"
            )

        with self.subTest("hemisphere length"):
            self.assertRaises(
                GPSError, self.gps_obj._validate_longitude, "11544.1007", "WE"
            )

        with self.subTest("hemisphere"):
            self.assertRaises(
                GPSError, self.gps_obj._validate_longitude, "11544.1007", "T"
            )

    def test_validate_elevation_fail(self):
        self.assertRaises(GPSError, self.gps_obj._validate_elevation, "kz")

    def test_validate_time_fail(self):
        with self.subTest("legth"):
            self.assertRaises(GPSError, self.gps_obj._validate_time, "0115")

        with self.subTest("type"):
            self.assertRaises(GPSError, self.gps_obj._validate_time, "0115SA")

    def test_validate_date_fail(self):
        with self.subTest("legth"):
            self.assertRaises(GPSError, self.gps_obj._validate_date, "0115")

        with self.subTest("type"):
            self.assertRaises(GPSError, self.gps_obj._validate_date, "0115SA")


class TestGPRMC(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gprmc_string = "GPRMC,183511,A,3443.6098,N,11544.1007,W,000.0,000.0,260919,013.1,E*"
        self.gps_obj = GPS(self.gprmc_string)

    def test_type(self):
        self.assertEqual(self.gps_obj.gps_type, "GPRMC")

    def test_latitude(self):
        self.assertAlmostEqual(self.gps_obj.latitude, 34.72683)

    def test_longitude(self):
        self.assertAlmostEqual(self.gps_obj.longitude, -115.73501166666667)

    def test_elevation(self):
        self.assertAlmostEqual(self.gps_obj.elevation, 0)

    def test_declination(self):
        self.assertAlmostEqual(self.gps_obj.declination, 13.1)

    def test_time_stamp(self):
        self.assertEqual(
            self.gps_obj.time_stamp.isoformat(), "2019-09-26T18:35:11"
        )


class TestGPGGA(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gpgga_string = (
            "GPGGA,183511,3443.6098,N,11544.1007,W,1,04,2.6,937.2,M,-28.1,M,*"
        )
        self.gps_obj = GPS(self.gpgga_string)

    def test_type(self):
        self.assertEqual(self.gps_obj.gps_type, "GPGGA")

    def test_latitude(self):
        self.assertAlmostEqual(self.gps_obj.latitude, 34.72683)

    def test_longitude(self):
        self.assertAlmostEqual(self.gps_obj.longitude, -115.73501166666667)

    def test_elevation(self):
        self.assertAlmostEqual(self.gps_obj.elevation, 937.2)

    def test_declination(self):
        self.assertEqual(self.gps_obj.declination, None)

    def test_time_stamp(self):
        self.assertEqual(
            self.gps_obj.time_stamp.isoformat(), "1980-01-01T18:35:11"
        )


# =============================================================================
# Run
# =============================================================================
if __name__ in "__main__":
    unittest.main()
