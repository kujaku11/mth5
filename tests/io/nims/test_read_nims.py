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
from collections import OrderedDict


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
        self.assertEqual(self.nims_obj.run_id, "mnp300a")

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

    def test_data_shape(self):
        self.assertEqual(self.nims_obj.ts_data.shape, (3357016, 5))

    def test_stamps_size(self):
        self.assertEqual(len(self.nims_obj.stamps), 402)

    def test_start(self):
        self.assertEqual(
            self.nims_obj.start_time, MTime("2019-09-26T18:33:21+00:00")
        )

    def test_end(self):
        self.assertEqual(
            self.nims_obj.end_time, MTime("2019-10-01T15:07:07.875000+00:00")
        )

    def test_latitude(self):
        self.assertAlmostEqual(self.nims_obj.latitude, 34.726826667)

    def test_longitude(self):
        self.assertAlmostEqual(self.nims_obj.longitude, -115.73501166)

    def test_elevation(self):
        self.assertAlmostEqual(self.nims_obj.elevation, 940.4)

    def test_declination(self):
        self.assertEqual(self.nims_obj.declination, 13.1)


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()),
    "Big data on local machine",
)
class TestNIMSToRunTS(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.runts = read_nims(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\nims\mnp300a.BIN"
        )
        self.maxDiff = None

    def test_station_metadata(self):
        station_metadata = OrderedDict(
            [
                ("acquired_by.name", None),
                ("channels_recorded", []),
                ("data_type", "BBMT"),
                ("geographic_name", "Budwieser Spring, CA, USA"),
                ("id", "mnp300"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 13.1),
                ("location.elevation", 940.4),
                ("location.latitude", 34.72682666666667),
                ("location.longitude", -115.73501166666667),
                ("orientation.method", None),
                ("orientation.reference_frame", "geomagnetic"),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.software.author", "none"),
                ("provenance.software.name", None),
                ("provenance.software.version", None),
                ("provenance.submitter.email", None),
                ("provenance.submitter.organization", None),
                ("run_list", ["mnp300a"]),
                ("time_period.end", "1980-01-01T00:00:00+00:00"),
                ("time_period.start", "1980-01-01T00:00:00+00:00"),
            ]
        )

        self.assertDictEqual(
            self.runts.station_metadata.to_dict(single=True), station_metadata
        )

    def test_run_metadata(self):
        run_metadata = OrderedDict(
            [
                ("channels_recorded_auxiliary", ["temperature"]),
                ("channels_recorded_electric", ["ex", "ey"]),
                ("channels_recorded_magnetic", ["hx", "hy", "hz"]),
                (
                    "comments",
                    "N/S: CRs: .764/.769 DCV: 3.5 ACV:1 \nE/W:  CRs: .930/.780 DCV: 28.2 ACV: 1\nHwy 40 800m S, X array",
                ),
                ("data_logger.firmware.author", "B. Narod"),
                ("data_logger.firmware.name", "nims"),
                ("data_logger.firmware.version", "1.0"),
                ("data_logger.id", "1105-3"),
                ("data_logger.manufacturer", "Narod"),
                ("data_logger.model", "1105-3"),
                ("data_logger.timing_system.drift", 0.0),
                ("data_logger.timing_system.type", "GPS"),
                ("data_logger.timing_system.uncertainty", 0.0),
                ("data_logger.type", "long period"),
                ("data_type", "MTLP"),
                ("id", "mnp300a"),
                ("sample_rate", 8.0),
                ("time_period.end", "2019-10-01T15:07:07.875000+00:00"),
                ("time_period.start", "2019-09-26T18:33:21+00:00"),
            ]
        )

        self.assertDictEqual(
            self.runts.run_metadata.to_dict(single=True), run_metadata
        )

    def test_ex_metadata(self):
        ex_metadata = OrderedDict(
            [
                ("channel_number", 4),
                ("component", "ex"),
                ("data_quality.rating.value", 0),
                ("dipole_length", 109.0),
                ("filter.applied", [False, False, False, False, False, False]),
                (
                    "filter.name",
                    [
                        "to_mt_units",
                        "dipole_109.00",
                        "nims_5_pole_butterworth",
                        "nims_1_pole_butterworth",
                        "e_analog_to_digital",
                        "ex_time_offset",
                    ],
                ),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("negative.elevation", 0.0),
                ("negative.id", "2"),
                ("negative.latitude", 0.0),
                ("negative.longitude", 0.0),
                ("negative.manufacturer", None),
                ("negative.type", None),
                ("positive.elevation", 0.0),
                ("positive.id", "1"),
                ("positive.latitude", 0.0),
                ("positive.longitude", 0.0),
                ("positive.manufacturer", None),
                ("positive.type", None),
                ("sample_rate", 8.0),
                ("time_period.end", "2019-10-01T15:07:08+00:00"),
                ("time_period.start", "2019-09-26T18:33:21+00:00"),
                ("type", "electric"),
                ("units", "counts"),
            ]
        )

        self.assertDictEqual(
            self.runts.ex.channel_metadata.to_dict(single=True), ex_metadata
        )

    def test_ey_metadata(self):
        ey_metadata = OrderedDict(
            [
                ("channel_number", 5),
                ("component", "ey"),
                ("data_quality.rating.value", 0),
                ("dipole_length", 101.0),
                ("filter.applied", [False, False, False, False, False, False]),
                (
                    "filter.name",
                    [
                        "to_mt_units",
                        "dipole_101.00",
                        "nims_5_pole_butterworth",
                        "nims_1_pole_butterworth",
                        "e_analog_to_digital",
                        "ey_time_offset",
                    ],
                ),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("negative.elevation", 0.0),
                ("negative.id", "4"),
                ("negative.latitude", 0.0),
                ("negative.longitude", 0.0),
                ("negative.manufacturer", None),
                ("negative.type", None),
                ("positive.elevation", 0.0),
                ("positive.id", "3"),
                ("positive.latitude", 0.0),
                ("positive.longitude", 0.0),
                ("positive.manufacturer", None),
                ("positive.type", None),
                ("sample_rate", 8.0),
                ("time_period.end", "2019-10-01T15:07:08+00:00"),
                ("time_period.start", "2019-09-26T18:33:21+00:00"),
                ("type", "electric"),
                ("units", "counts"),
            ]
        )

        self.assertDictEqual(
            self.runts.ey.channel_metadata.to_dict(single=True), ey_metadata
        )

    def test_hx_metadata(self):
        hx_metadata = OrderedDict(
            [
                ("channel_number", 1),
                ("component", "hx"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [False, False, False]),
                (
                    "filter.name",
                    [
                        "nims_3_pole_butterworth",
                        "h_analog_to_digital",
                        "hx_time_offset",
                    ],
                ),
                ("location.elevation", 0.0),
                ("location.latitude", 0.0),
                ("location.longitude", 0.0),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 8.0),
                ("sensor.id", "1305-3"),
                ("sensor.manufacturer", "Barry Narod"),
                ("sensor.type", "fluxgate triaxial magnetometer"),
                ("time_period.end", "2019-10-01T15:07:08+00:00"),
                ("time_period.start", "2019-09-26T18:33:21+00:00"),
                ("type", "magnetic"),
                ("units", "counts"),
            ]
        )

        self.assertDictEqual(
            self.runts.hx.channel_metadata.to_dict(single=True), hx_metadata
        )

    def test_hy_metadata(self):
        hy_metadata = OrderedDict(
            [
                ("channel_number", 2),
                ("component", "hy"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [False, False, False]),
                (
                    "filter.name",
                    [
                        "nims_3_pole_butterworth",
                        "h_analog_to_digital",
                        "hy_time_offset",
                    ],
                ),
                ("location.elevation", 0.0),
                ("location.latitude", 0.0),
                ("location.longitude", 0.0),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 8.0),
                ("sensor.id", "1305-3"),
                ("sensor.manufacturer", "Barry Narod"),
                ("sensor.type", "fluxgate triaxial magnetometer"),
                ("time_period.end", "2019-10-01T15:07:08+00:00"),
                ("time_period.start", "2019-09-26T18:33:21+00:00"),
                ("type", "magnetic"),
                ("units", "counts"),
            ]
        )

        self.assertDictEqual(
            self.runts.hy.channel_metadata.to_dict(single=True), hy_metadata
        )

    def test_hz_metadata(self):
        hz_metadata = OrderedDict(
            [
                ("channel_number", 3),
                ("component", "hz"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [False, False, False]),
                (
                    "filter.name",
                    [
                        "nims_3_pole_butterworth",
                        "h_analog_to_digital",
                        "hz_time_offset",
                    ],
                ),
                ("location.elevation", 0.0),
                ("location.latitude", 0.0),
                ("location.longitude", 0.0),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 90.0),
                ("sample_rate", 8.0),
                ("sensor.id", "1305-3"),
                ("sensor.manufacturer", "Barry Narod"),
                ("sensor.type", "fluxgate triaxial magnetometer"),
                ("time_period.end", "2019-10-01T15:07:08+00:00"),
                ("time_period.start", "2019-09-26T18:33:21+00:00"),
                ("type", "magnetic"),
                ("units", "counts"),
            ]
        )

        self.assertDictEqual(
            self.runts.hz.channel_metadata.to_dict(single=True), hz_metadata
        )

    def test_temperature_metadata(self):
        t_metadata = OrderedDict(
            [
                ("channel_number", 6),
                ("component", "temperature"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [False]),
                ("filter.name", []),
                ("location.elevation", 0.0),
                ("location.latitude", 0.0),
                ("location.longitude", 0.0),
                ("measurement_azimuth", 0.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 8.0),
                ("sensor.id", None),
                ("sensor.manufacturer", None),
                ("sensor.type", None),
                ("time_period.end", "2019-10-01T15:07:08+00:00"),
                ("time_period.start", "2019-09-26T18:33:21+00:00"),
                ("type", "auxiliary"),
                ("units", "celsius"),
            ]
        )

        self.assertDictEqual(
            self.runts.temperature.channel_metadata.to_dict(single=True),
            t_metadata,
        )

    def test_dataset_dims(self):
        self.assertEqual(self.runts.dataset.dims["time"], 3357016)

    def test_calibrate(self):
        calibrated_run = self.runts.calibrate()

        for comp in ["hx", "hy", "hz"]:
            ch = getattr(calibrated_run, comp)
            with self.subTest("units"):
                self.assertEqual(ch.channel_metadata.units, "nT")
            with self.subTest("applied"):
                self.assertListEqual(
                    ch.channel_metadata.filter.applied, [True, True, True]
                )

        for comp in ["ex", "ey"]:
            ch = getattr(calibrated_run, comp)
            with self.subTest("units"):
                self.assertEqual(ch.channel_metadata.units, "mV/km")
            with self.subTest("applied"):
                self.assertListEqual(
                    ch.channel_metadata.filter.applied,
                    [True, True, True, True, True, True],
                )

        ch = getattr(calibrated_run, "temperature")
        with self.subTest("units"):
            self.assertEqual(ch.channel_metadata.units, "celsius")
        with self.subTest("applied"):
            self.assertListEqual(
                ch.channel_metadata.filter.applied,
                [True],
            )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
