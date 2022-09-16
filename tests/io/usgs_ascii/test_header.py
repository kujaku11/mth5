# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:47:42 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path
from collections import OrderedDict

from mth5.io.usgs_ascii import AsciiMetadata

# =============================================================================


@unittest.skipIf("peacock" not in str(Path(__file__).as_posix()), "local file")
class TestAsciiMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.header = AsciiMetadata(
            fn=r"C:\Users\jpeacock\OneDrive - DOI\mt\usgs_ascii\rgr003a_converted.asc"
        )
        self.header.read_metadata()
        self.maxDiff = None

    def test_fn(self):
        self.assertIsInstance(self.header.fn, Path)

    def test_coordinate_system(self):
        self.assertEqual(self.header.coordinate_system, "geographic north")

    def test_elevation(self):
        self.assertEqual(self.header.elevation, 1803.0)

    def test_missing_data_flag(self):
        self.assertEqual(self.header.missing_data_flag, "1.000e+09")

    def test_end(self):
        self.assertEqual(self.header.end, "2012-08-24T16:25:26+00:00")

    def test_ex(self):
        self.assertDictEqual(
            self.header.ex_metadata.to_dict(single=True),
            OrderedDict(
                [
                    ("channel_number", 32),
                    ("component", None),
                    ("data_quality.rating.value", 0),
                    ("dipole_length", 100.0),
                    ("filter.applied", [False]),
                    ("filter.name", []),
                    ("measurement_azimuth", 9.0),
                    ("measurement_tilt", 0.0),
                    ("negative.elevation", 0.0),
                    ("negative.id", None),
                    ("negative.latitude", 0.0),
                    ("negative.longitude", 0.0),
                    ("negative.manufacturer", None),
                    ("negative.type", None),
                    ("positive.elevation", 0.0),
                    ("positive.id", None),
                    ("positive.latitude", 0.0),
                    ("positive.longitude", 0.0),
                    ("positive.manufacturer", None),
                    ("positive.type", None),
                    ("sample_rate", 4.0),
                    ("time_period.end", "2012-08-24T16:25:26+00:00"),
                    ("time_period.start", "2012-08-21T22:02:27+00:00"),
                    ("type", "electric"),
                    ("units", None),
                ]
            ),
        )

    def test_ey(self):
        self.assertDictEqual(
            self.header.ey_metadata.to_dict(single=True),
            OrderedDict(
                [
                    ("channel_number", 34),
                    ("component", None),
                    ("data_quality.rating.value", 0),
                    ("dipole_length", 102.0),
                    ("filter.applied", [False]),
                    ("filter.name", []),
                    ("measurement_azimuth", 99.0),
                    ("measurement_tilt", 0.0),
                    ("negative.elevation", 0.0),
                    ("negative.id", None),
                    ("negative.latitude", 0.0),
                    ("negative.longitude", 0.0),
                    ("negative.manufacturer", None),
                    ("negative.type", None),
                    ("positive.elevation", 0.0),
                    ("positive.id", None),
                    ("positive.latitude", 0.0),
                    ("positive.longitude", 0.0),
                    ("positive.manufacturer", None),
                    ("positive.type", None),
                    ("sample_rate", 4.0),
                    ("time_period.end", "2012-08-24T16:25:26+00:00"),
                    ("time_period.start", "2012-08-21T22:02:27+00:00"),
                    ("type", "electric"),
                    ("units", None),
                ]
            ),
        )

    def test_hx(self):
        self.assertDictEqual(
            self.header.hx_metadata.to_dict(single=True),
            OrderedDict(
                [
                    ("channel_number", 31),
                    ("component", None),
                    ("data_quality.rating.value", 0),
                    ("filter.applied", [False]),
                    ("filter.name", []),
                    ("location.elevation", 0.0),
                    ("location.latitude", 0.0),
                    ("location.longitude", 0.0),
                    ("measurement_azimuth", 9.0),
                    ("measurement_tilt", 0.0),
                    ("sample_rate", 4.0),
                    ("sensor.id", "2311-11"),
                    ("sensor.manufacturer", None),
                    ("sensor.type", None),
                    ("time_period.end", "2012-08-24T16:25:26+00:00"),
                    ("time_period.start", "2012-08-21T22:02:27+00:00"),
                    ("type", "magnetic"),
                    ("units", None),
                ]
            ),
        )

    def test_hy(self):
        self.assertDictEqual(
            self.header.hy_metadata.to_dict(single=True),
            OrderedDict(
                [
                    ("channel_number", 33),
                    ("component", None),
                    ("data_quality.rating.value", 0),
                    ("filter.applied", [False]),
                    ("filter.name", []),
                    ("location.elevation", 0.0),
                    ("location.latitude", 0.0),
                    ("location.longitude", 0.0),
                    ("measurement_azimuth", 99.0),
                    ("measurement_tilt", 0.0),
                    ("sample_rate", 4.0),
                    ("sensor.id", "2311-11"),
                    ("sensor.manufacturer", None),
                    ("sensor.type", None),
                    ("time_period.end", "2012-08-24T16:25:26+00:00"),
                    ("time_period.start", "2012-08-21T22:02:27+00:00"),
                    ("type", "magnetic"),
                    ("units", None),
                ]
            ),
        )

    def test_hz(self):
        self.assertDictEqual(
            self.header.hz_metadata.to_dict(single=True),
            OrderedDict(
                [
                    ("channel_number", 35),
                    ("component", None),
                    ("data_quality.rating.value", 0),
                    ("filter.applied", [False]),
                    ("filter.name", []),
                    ("location.elevation", 0.0),
                    ("location.latitude", 0.0),
                    ("location.longitude", 0.0),
                    ("measurement_azimuth", 0.0),
                    ("measurement_tilt", 0.0),
                    ("sample_rate", 4.0),
                    ("sensor.id", "2311-11"),
                    ("sensor.manufacturer", None),
                    ("sensor.type", None),
                    ("time_period.end", "2012-08-24T16:25:26+00:00"),
                    ("time_period.start", "2012-08-21T22:02:27+00:00"),
                    ("type", "magnetic"),
                    ("units", None),
                ]
            ),
        )

    def test_latitude(self):
        self.assertEqual(self.header.latitude, 39.282)

    def test_longitude(self):
        self.assertEqual(self.header.longitude, -108.1582)

    def test_n_channels(self):
        self.assertEqual(self.header.n_channels, 5)

    def test_n_samples(self):
        self.assertEqual(self.header.n_samples, 955916)

    def test_run_id(self):
        self.assertEqual(self.header.run_id, "rgr003a")

    def test_run_metadata(self):
        self.assertDictEqual(
            self.header.run_metadata.to_dict(single=True),
            OrderedDict(
                [
                    ("channels_recorded_auxiliary", []),
                    ("channels_recorded_electric", []),
                    ("channels_recorded_magnetic", []),
                    ("data_logger.firmware.author", None),
                    ("data_logger.firmware.name", None),
                    ("data_logger.firmware.version", None),
                    ("data_logger.id", "2311-11"),
                    ("data_logger.manufacturer", None),
                    ("data_logger.timing_system.drift", 0.0),
                    ("data_logger.timing_system.type", "GPS"),
                    ("data_logger.timing_system.uncertainty", 0.0),
                    ("data_logger.type", None),
                    ("data_type", "BBMT"),
                    ("id", None),
                    ("sample_rate", 4.0),
                    ("time_period.end", "2012-08-24T16:25:26+00:00"),
                    ("time_period.start", "2012-08-21T22:02:27+00:00"),
                ]
            ),
        )

    def test_sample_rate(self):
        self.assertEqual(self.header.sample_rate, 4.0)

    def test_site_id(self):
        self.assertEqual(self.header.site_id, "003")

    def test_start(self):
        self.assertEqual(self.header.start, "2012-08-21T22:02:27+00:00")

    def test_station_metadata(self):
        self.assertDictEqual(
            self.header.station_metadata.to_dict(single=True),
            OrderedDict(
                [
                    ("acquired_by.name", None),
                    ("channels_recorded", []),
                    ("data_type", "BBMT"),
                    ("geographic_name", None),
                    ("id", "003"),
                    ("location.declination.model", "WMM"),
                    ("location.declination.value", 0.0),
                    ("location.elevation", 0.0),
                    ("location.latitude", 39.282),
                    ("location.longitude", -108.1582),
                    ("orientation.method", None),
                    ("orientation.reference_frame", "geographic"),
                    ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                    ("provenance.software.author", "none"),
                    ("provenance.software.name", None),
                    ("provenance.software.version", None),
                    ("provenance.submitter.email", None),
                    ("provenance.submitter.organization", None),
                    ("run_list", []),
                    ("time_period.end", "2012-08-24T16:25:26+00:00"),
                    ("time_period.start", "2012-08-21T22:02:27+00:00"),
                ]
            ),
        )

    def test_survey_id(self):
        self.assertEqual(self.header.survey_id, "RGR")

    def test_survey_metadata(self):
        self.assertDictEqual(
            self.header.survey_metadata.to_dict(single=True),
            OrderedDict(
                [
                    ("citation_dataset.doi", None),
                    ("citation_journal.doi", None),
                    ("country", None),
                    ("datum", "WGS84"),
                    ("geographic_name", None),
                    ("id", "RGR"),
                    ("name", None),
                    ("northwest_corner.latitude", 0.0),
                    ("northwest_corner.longitude", 0.0),
                    ("project", None),
                    ("project_lead.email", None),
                    ("project_lead.organization", None),
                    ("release_license", "CC-0"),
                    ("southeast_corner.latitude", 0.0),
                    ("southeast_corner.longitude", 0.0),
                    ("summary", None),
                    ("time_period.end_date", "1980-01-01"),
                    ("time_period.start_date", "1980-01-01"),
                ]
            ),
        )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
