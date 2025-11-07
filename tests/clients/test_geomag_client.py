# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:33:20 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from collections import OrderedDict

import numpy as np

from mth5.clients.geomag import GeomagClient


# =============================================================================


class TestGeomagClient(unittest.TestCase):
    def setUp(self):
        self.client = GeomagClient()

    def test_observatory(self):
        self.client.observatory = "frn"
        self.assertEqual("FRN", self.client.observatory)

    def test_observatory_fail(self):
        def set_observatory(value):
            self.client.observatory = value

        with self.subTest("ValueError"):
            self.assertRaises(ValueError, set_observatory, "ten")

        with self.subTest("TypeError"):
            self.assertRaises(TypeError, set_observatory, 10)

    def test_elements(self):
        self.client.elements = "x"
        self.assertListEqual(["X"], self.client.elements)

    def test_elements_from_string(self):
        self.client.elements = "x,y"
        self.assertListEqual(["X", "Y"], self.client.elements)

    def test_elements_fail_input(self):
        def set_elements(value):
            self.client.elements = value

        with self.subTest("ValueError"):
            self.assertRaises(ValueError, set_elements, "v")

        with self.subTest("ValueError_list"):
            self.assertRaises(ValueError, set_elements, ["x", "v"])

        with self.subTest("TypeError"):
            self.assertRaises(TypeError, set_elements, 10)

        with self.subTest("TypeError_list"):
            self.assertRaises(TypeError, set_elements, ["x", 10])

    def test_sampling_period(self):
        self.client.sampling_period = "1"
        self.assertEqual(1, self.client.sampling_period)

    def test_sampling_period_fail(self):
        def set_period(value):
            self.client.sampling_period = value

        with self.subTest("ValueError_bad_number"):
            self.assertRaises(ValueError, set_period, "p")

        with self.subTest("TypeError"):
            self.assertRaises(TypeError, set_period, [1])

        with self.subTest("not in list"):
            self.assertRaises(ValueError, set_period, 10)

    def test_start(self):
        self.client.start = "2020-01-01T00:00:00+00:00"
        self.assertEqual(self.client.start, "2020-01-01T00:00:00Z")

    def test_end(self):
        self.client.end = "2020-01-01T00:00:00+00:00"
        self.assertEqual(self.client.end, "2020-01-01T00:00:00Z")

    def test_estimate_chunks(self):
        self.client.start = "2021-04-05T00:00:00+00:00"
        self.client.end = "2021-04-16T00:00:00+00:00"

        self.assertListEqual(
            [
                ("2021-04-05T00:00:00Z", "2021-04-07T00:00:00Z"),
                ("2021-04-07T00:00:00Z", "2021-04-09T00:00:00Z"),
                ("2021-04-09T00:00:00Z", "2021-04-11T00:00:00Z"),
                ("2021-04-11T00:00:00Z", "2021-04-13T00:00:00Z"),
                ("2021-04-13T00:00:00Z", "2021-04-15T00:00:00Z"),
                ("2021-04-15T00:00:00Z", "2021-04-16T00:00:00Z"),
            ],
            self.client.get_chunks(),
        )

    def test_get_request_params(self):
        self.client.observatory = "frn"
        self.client.type = "adjusted"
        self.client.elements = ["x", "y"]
        self.sampling_period = 1

        start = "2020-01-01T00:00:00+00:00"
        end = "2020-01-02T12:00:00+00:00"

        request_dict = {
            "id": self.client.observatory,
            "type": self.client.type,
            "elements": ",".join(self.client.elements),
            "sampling_period": self.client.sampling_period,
            "format": "json",
            "starttime": start,
            "endtime": end,
        }

        self.assertDictEqual(request_dict, self.client._get_request_params(start, end))


class TestGeomagClientGetData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.client = GeomagClient()
        self.client.observatory = "frn"
        self.client.sampling_period = 1
        self.client.start = "2020-01-01T00:00:00"
        self.client.end = "2020-01-01T12:00:00"
        self.client.elements = ["x", "y"]

        self.maxDiff = None

        try:
            self.r = self.client.get_data()
        except IOError as error:
            self.skipTest(error)

    def test_survey_metadata(self):
        s = OrderedDict(
            [
                ("citation_dataset.doi", None),
                ("citation_journal.doi", None),
                ("datum", "WGS84"),
                ("geographic_name", None),
                ("id", "USGS-GEOMAG"),
                ("name", None),
                ("northwest_corner.latitude", 0.0),
                ("northwest_corner.longitude", 0.0),
                ("project", None),
                ("project_lead.email", None),
                ("project_lead.organization", None),
                ("release_license", "CC0-1.0"),
                ("southeast_corner.latitude", 0.0),
                ("southeast_corner.longitude", 0.0),
                ("summary", None),
                ("time_period.end_date", "2020-01-01"),
                ("time_period.start_date", "2020-01-01"),
            ]
        )

        for key, value in s.items():
            if key not in ["mth5_type", "hdf5_reference"]:
                with self.subTest(key):
                    self.assertEqual(
                        value, self.r.survey_metadata.get_attr_from_name(key)
                    )

    def test_station_metadata(self):
        s = OrderedDict(
            [
                ("acquired_by.name", None),
                ("channels_recorded", ["hx", "hy"]),
                ("data_type", "BBMT"),
                ("fdsn.id", "FRN"),
                ("geographic_name", None),
                ("id", "Fresno"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 0.0),
                ("location.elevation", 331.0),
                ("location.latitude", 37.091),
                ("location.longitude", -119.71799999999999),
                ("orientation.method", None),
                ("orientation.reference_frame", "geographic"),
                ("provenance.creation_time", "2023-03-20T23:02:17+00:00"),
                ("provenance.software.author", None),
                ("provenance.software.name", None),
                ("provenance.software.version", None),
                ("provenance.submitter.email", None),
                ("provenance.submitter.organization", None),
                ("release_license", "CC0-1.0"),
                ("run_list", ["001"]),
                ("time_period.end", "2020-01-01T12:00:00+00:00"),
                ("time_period.start", "2020-01-01T00:00:00+00:00"),
            ]
        )

        for key, value in s.items():
            with self.subTest(key):
                r_value = self.r.station_metadata.get_attr_from_name(key)
                if key in ["provenance.creation_time"]:
                    self.assertNotEqual(value, r_value)

                elif key in ["mth5_type", "hdf5_reference"]:
                    pass
                else:
                    self.assertEqual(value, r_value)

    def test_run_metadata(self):
        r = OrderedDict(
            [
                ("channels_recorded_auxiliary", []),
                ("channels_recorded_electric", []),
                ("channels_recorded_magnetic", ["hx", "hy"]),
                ("data_logger.firmware.author", None),
                ("data_logger.firmware.name", None),
                ("data_logger.firmware.version", None),
                ("data_logger.id", None),
                ("data_logger.manufacturer", None),
                ("data_logger.timing_system.drift", 0.0),
                ("data_logger.timing_system.type", "GPS"),
                ("data_logger.timing_system.uncertainty", 0.0),
                ("data_logger.type", None),
                ("data_type", "BBMT"),
                ("id", "001"),
                ("sample_rate", 1.0),
                ("time_period.end", "2020-01-01T12:00:00+00:00"),
                ("time_period.start", "2020-01-01T00:00:00+00:00"),
            ]
        )

        for key, value in r.items():
            if key not in ["mth5_type", "hdf5_reference"]:
                with self.subTest(key):
                    self.assertEqual(value, self.r.run_metadata.get_attr_from_name(key))

    def test_hx_metadata(self):
        ch = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "hx"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [True]),
                ("filter.name", []),
                ("location.elevation", 331.0),
                ("location.latitude", 37.091),
                ("location.longitude", -119.71799999999999),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 1.0),
                ("sensor.id", None),
                ("sensor.manufacturer", None),
                ("sensor.type", None),
                ("time_period.end", "2020-01-01T12:00:00+00:00"),
                ("time_period.start", "2020-01-01T00:00:00+00:00"),
                ("type", "magnetic"),
                ("units", "nanotesla"),
            ]
        )

        for key, value in ch.items():
            if key not in ["mth5_type", "hdf5_reference"]:
                with self.subTest(key):
                    self.assertEqual(
                        value,
                        self.r.hx.channel_metadata.get_attr_from_name(key),
                    )

    def test_hy_metadata(self):
        ch = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "hy"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [True]),
                ("filter.name", []),
                ("location.elevation", 331.0),
                ("location.latitude", 37.091),
                ("location.longitude", -119.71799999999999),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 1.0),
                ("sensor.id", None),
                ("sensor.manufacturer", None),
                ("sensor.type", None),
                ("time_period.end", "2020-01-01T12:00:00+00:00"),
                ("time_period.start", "2020-01-01T00:00:00+00:00"),
                ("type", "magnetic"),
                ("units", "nanotesla"),
            ]
        )

        for key, value in ch.items():
            if key not in ["mth5_type", "hdf5_reference"]:
                with self.subTest(key):
                    self.assertEqual(
                        value,
                        self.r.hy.channel_metadata.get_attr_from_name(key),
                    )

    def test_hx_data(self):
        hx = np.array(
            [
                22627.05,
                22627.04,
                22627.053,
                22627.044,
                22627.051,
                22627.05,
                22627.074,
                22627.087,
                22627.048,
                22627.08,
            ]
        )
        self.assertTrue(np.allclose(hx, self.r.hx.ts[0:10]))

    def test_hy_data(self):
        hy = np.array(
            [
                5111.451,
                5111.438,
                5111.44,
                5111.441,
                5111.44,
                5111.425,
                5111.482,
                5111.495,
                5111.452,
                5111.442,
            ]
        )
        self.assertTrue(np.allclose(hy, self.r.hy.ts[0:10]))


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
