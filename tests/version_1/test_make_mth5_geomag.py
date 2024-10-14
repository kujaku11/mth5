# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:40:28 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import unittest
from collections import OrderedDict
from pathlib import Path
import pandas as pd

from mth5.clients import MakeMTH5

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()),
    "Downloading takes too long",
)
class TestMakeMTH5FromGeomag(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # self.make_mth5 = MakeMTH5(
        #     mth5_version="0.1.0", interact=True, save_path=Path().cwd()
        # )

        self.request_df = pd.DataFrame(
            {
                "observatory": ["frn", "frn", "ott", "ott"],
                "type": ["adjusted"] * 4,
                "elements": [["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"]],
                "sampling_period": [1, 1, 1, 1],
                "start": [
                    "2022-01-01T00:00:00",
                    "2022-01-03T00:00:00",
                    "2022-01-01T00:00:00",
                    "2022-01-03T00:00:00",
                ],
                "end": [
                    "2022-01-02T00:00:00",
                    "2022-01-04T00:00:00",
                    "2022-01-02T00:00:00",
                    "2022-01-04T00:00:00",
                ],
            }
        )

        # self.m = self.make_mth5.from_usgs_geomag(self.request_df)
        self.m = MakeMTH5.from_usgs_geomag(
            self.request_df,
            mth5_version="0.1.0",
            interact=True,
            save_path=Path().cwd(),
        )

    def test_file_exists(self):
        self.assertTrue(self.m.filename.exists())

    def test_file_version(self):
        self.assertEqual(self.m.file_version, "0.1.0")

    def test_survey_metadata(self):
        sm = OrderedDict(
            [
                ("citation_dataset.doi", None),
                ("citation_journal.doi", None),
                ("datum", "WGS84"),
                ("geographic_name", None),
                ("hdf5_reference", "<HDF5 object reference>"),
                ("id", "USGS-GEOMAG"),
                ("mth5_type", "Survey"),
                ("name", None),
                ("northwest_corner.latitude", 45.4),
                ("northwest_corner.longitude", -119.71799999999999),
                ("project", None),
                ("project_lead.email", None),
                ("project_lead.organization", None),
                ("release_license", "CC0-1.0"),
                ("southeast_corner.latitude", 37.091),
                ("southeast_corner.longitude", -75.5),
                ("summary", None),
                ("time_period.end_date", "2022-01-04"),
                ("time_period.start_date", "2022-01-01"),
            ]
        )

        for key, value in sm.items():
            with self.subTest(key):
                r_value = self.m.survey_group.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    def test_station_frn(self):
        s = self.m.get_station("Fresno")
        sm = OrderedDict(
            [
                ("acquired_by.name", None),
                ("channels_recorded", ["hx", "hy"]),
                ("data_type", "BBMT"),
                ("fdsn.id", "FRN"),
                ("geographic_name", None),
                ("hdf5_reference", "<HDF5 object reference>"),
                ("id", "Fresno"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 0.0),
                ("location.elevation", 331.0),
                ("location.latitude", 37.091),
                ("location.longitude", -119.71799999999999),
                ("mth5_type", "Station"),
                ("orientation.method", None),
                ("orientation.reference_frame", "geographic"),
                ("provenance.creation_time", "2023-03-22T21:17:53+00:00"),
                ("provenance.software.author", None),
                ("provenance.software.name", None),
                ("provenance.software.version", None),
                ("provenance.submitter.email", None),
                ("provenance.submitter.organization", None),
                ("release_license", "CC0-1.0"),
                ("run_list", ["sp1_001", "sp1_002"]),
                ("time_period.end", "2022-01-04T00:00:00+00:00"),
                ("time_period.start", "2022-01-01T00:00:00+00:00"),
            ]
        )

        for key, value in sm.items():
            with self.subTest(key):
                r_value = s.metadata.get_attr_from_name(key)
                if key in ["provenance.creation_time"]:
                    self.assertNotEqual(value, r_value)

                elif key in ["mth5_type", "hdf5_reference"]:
                    pass
                else:
                    self.assertEqual(value, r_value)

    def test_fresno_run_001_metadata(self):
        r = self.m.get_run("Fresno", "sp1_001")
        rm = OrderedDict(
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
                ("hdf5_reference", "<HDF5 object reference>"),
                ("id", "sp1_001"),
                ("mth5_type", "Run"),
                ("sample_rate", 1.0),
                ("time_period.end", "2022-01-02T00:00:00+00:00"),
                ("time_period.start", "2022-01-01T00:00:00+00:00"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                r_value = r.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    def test_fresno_run_002_metadata(self):
        r = self.m.get_run("Fresno", "sp1_002")
        rm = OrderedDict(
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
                ("hdf5_reference", "<HDF5 object reference>"),
                ("id", "sp1_002"),
                ("mth5_type", "Run"),
                ("sample_rate", 1.0),
                ("time_period.end", "2022-01-04T00:00:00+00:00"),
                ("time_period.start", "2022-01-03T00:00:00+00:00"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                r_value = r.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    def test_fresno_run_001_hx_metadata(self):
        r = self.m.get_channel("Fresno", "sp1_001", "hx")
        rm = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "hx"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [True]),
                ("filter.name", []),
                ("hdf5_reference", "<HDF5 object reference>"),
                ("location.elevation", 331.0),
                ("location.latitude", 37.091),
                ("location.longitude", -119.71799999999999),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("mth5_type", "Magnetic"),
                ("sample_rate", 1.0),
                ("sensor.id", None),
                ("sensor.manufacturer", None),
                ("sensor.type", None),
                ("time_period.end", "2022-01-02T00:00:00+00:00"),
                ("time_period.start", "2022-01-01T00:00:00+00:00"),
                ("type", "magnetic"),
                ("units", "nanotesla"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                r_value = r.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    def test_fresno_run_001_hy_metadata(self):
        r = self.m.get_channel("Fresno", "sp1_001", "hy")
        rm = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "hy"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [True]),
                ("filter.name", []),
                ("hdf5_reference", "<HDF5 object reference>"),
                ("location.elevation", 331.0),
                ("location.latitude", 37.091),
                ("location.longitude", -119.71799999999999),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("mth5_type", "Magnetic"),
                ("sample_rate", 1.0),
                ("sensor.id", None),
                ("sensor.manufacturer", None),
                ("sensor.type", None),
                ("time_period.end", "2022-01-02T00:00:00+00:00"),
                ("time_period.start", "2022-01-01T00:00:00+00:00"),
                ("type", "magnetic"),
                ("units", "nanotesla"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                r_value = r.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    def test_station_ottowa(self):
        s = self.m.get_station("Ottowa")
        sm = OrderedDict(
            [
                ("acquired_by.name", None),
                ("channels_recorded", ["hx", "hy"]),
                ("data_type", "BBMT"),
                ("fdsn.id", "OTT"),
                ("geographic_name", None),
                ("hdf5_reference", "<HDF5 object reference>"),
                ("id", "Ottowa"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 0.0),
                ("location.elevation", 0.0),
                ("location.latitude", 45.4),
                ("location.longitude", -75.5),
                ("mth5_type", "Station"),
                ("orientation.method", None),
                ("orientation.reference_frame", "geographic"),
                ("provenance.creation_time", "2023-03-22T21:18:01+00:00"),
                ("provenance.software.author", None),
                ("provenance.software.name", None),
                ("provenance.software.version", None),
                ("provenance.submitter.email", None),
                ("provenance.submitter.organization", None),
                ("release_license", "CC0-1.0"),
                ("run_list", ["sp1_001", "sp1_002"]),
                ("time_period.end", "2022-01-04T00:00:00+00:00"),
                ("time_period.start", "2022-01-01T00:00:00+00:00"),
            ]
        )

        for key, value in sm.items():
            with self.subTest(key):
                r_value = s.metadata.get_attr_from_name(key)
                if key in ["provenance.creation_time"]:
                    self.assertNotEqual(value, r_value)

                elif key in ["mth5_type", "hdf5_reference"]:
                    pass
                else:
                    self.assertEqual(value, r_value)

    def test_ottowa_run_001_metadata(self):
        r = self.m.get_run("Ottowa", "sp1_001")
        rm = OrderedDict(
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
                ("hdf5_reference", "<HDF5 object reference>"),
                ("id", "sp1_001"),
                ("mth5_type", "Run"),
                ("sample_rate", 1.0),
                ("time_period.end", "2022-01-02T00:00:00+00:00"),
                ("time_period.start", "2022-01-01T00:00:00+00:00"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                r_value = r.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    def test_ottowa_run_002_metadata(self):
        r = self.m.get_run("Ottowa", "sp1_002")
        rm = OrderedDict(
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
                ("hdf5_reference", "<HDF5 object reference>"),
                ("id", "sp1_002"),
                ("mth5_type", "Run"),
                ("sample_rate", 1.0),
                ("time_period.end", "2022-01-04T00:00:00+00:00"),
                ("time_period.start", "2022-01-03T00:00:00+00:00"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                r_value = r.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    def test_ottowa_run_001_hx_metadata(self):
        r = self.m.get_channel("Ottowa", "sp1_001", "hx")
        rm = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "hx"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [True]),
                ("filter.name", []),
                ("hdf5_reference", "<HDF5 object reference>"),
                ("location.elevation", 0.0),
                ("location.latitude", 45.4),
                ("location.longitude", -75.5),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("mth5_type", "Magnetic"),
                ("sample_rate", 1.0),
                ("sensor.id", None),
                ("sensor.manufacturer", None),
                ("sensor.type", None),
                ("time_period.end", "2022-01-02T00:00:00+00:00"),
                ("time_period.start", "2022-01-01T00:00:00+00:00"),
                ("type", "magnetic"),
                ("units", "nanotesla"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                r_value = r.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    def test_ottowa_run_001_hy_metadata(self):
        r = self.m.get_channel("Ottowa", "sp1_001", "hy")
        rm = OrderedDict(
            [
                ("channel_number", 0),
                ("component", "hy"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [True]),
                ("filter.name", []),
                ("hdf5_reference", "<HDF5 object reference>"),
                ("location.elevation", 0.0),
                ("location.latitude", 45.4),
                ("location.longitude", -75.5),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("mth5_type", "Magnetic"),
                ("sample_rate", 1.0),
                ("sensor.id", None),
                ("sensor.manufacturer", None),
                ("sensor.type", None),
                ("time_period.end", "2022-01-02T00:00:00+00:00"),
                ("time_period.start", "2022-01-01T00:00:00+00:00"),
                ("type", "magnetic"),
                ("units", "nanotesla"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                r_value = r.metadata.get_attr_from_name(key)
                if key not in ["mth5_type", "hdf5_reference"]:
                    self.assertEqual(value, r_value)

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.m.filename.unlink()


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
