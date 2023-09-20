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
from collections import OrderedDict

import numpy as np

from mth5.io.zen import Z3D

from mt_metadata.timeseries.filters import (
    ChannelResponseFilter,
    FrequencyResponseTableFilter,
    CoefficientFilter,
)

# =============================================================================


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()), "local files"
)
class TestZ3DEY(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = Path(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\example_z3d_data\bm100_20220517_131017_256_EY.Z3D"
        )
        self.z3d = Z3D(self.fn)
        self.z3d.read_z3d()

        self.maxDiff = None

    def test_fn(self):
        self.assertEqual(self.fn, self.z3d.fn)

    def test_file_size(self):
        self.assertEqual(self.z3d.file_size, 10759100)

    def test_n_samples(self):
        self.assertEqual(self.z3d.n_samples, 2530304)

    def test_station(self):
        self.assertEqual(self.z3d.station, "100")

    def test_dipole_length(self):
        self.assertEqual(self.z3d.dipole_length, 56.0)

    def test_azimuth(self):
        self.assertEqual(self.z3d.azimuth, 0)

    def test_component(self):
        self.assertEqual(self.z3d.component, "ey")

    def test_latitude(self):
        self.assertAlmostEqual(self.z3d.latitude, 40.49757833327694)

    def test_longitude(self):
        self.assertAlmostEqual(self.z3d.longitude, -116.8211900230401)

    def test_elevation(self):
        self.assertAlmostEqual(self.z3d.elevation, 1456.3)

    def test_gps_stamps(self):
        self.assertEqual(9885, self.z3d.gps_stamps.size)

    def test_sample_rate(self):
        self.assertEqual(self.z3d.sample_rate, 256)

    def test_start(self):
        self.assertEqual(self.z3d.start, "2022-05-17T13:09:58+00:00")

    def test_end(self):
        self.assertEqual(self.z3d.end, "2022-05-17T15:54:42+00:00")

    def test_zen_schedule(self):
        self.assertEqual(self.z3d.zen_schedule, "2022-05-17T13:09:58+00:00")

    def test_coil_number(self):
        self.assertEqual(self.z3d.coil_number, None)

    def test_channel_number(self):
        self.assertEqual(self.z3d.channel_number, 5)

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

    def test_get_gps_time(self):
        self.assertTupleEqual(
            self.z3d.get_gps_time(220216, 2210), (215.056, 2210.0)
        )

    def test_get_utc_date_time(self):
        self.assertEqual(
            self.z3d.get_UTC_date_time(2210, 220216),
            "2022-05-17T13:09:58+00:00",
        )

    def test_channel_metadata(self):
        ey = OrderedDict(
            [
                ("ac.end", 3.1080129002282815e-05),
                ("ac.start", 3.4870685796509496e-05),
                ("channel_number", 5),
                ("component", "ey"),
                ("data_quality.rating.value", 0),
                ("dc.end", 0.019371436521409924),
                ("dc.start", 0.019130984313785026),
                ("dipole_length", 56.0),
                ("filter.applied", [False, False]),
                (
                    "filter.name",
                    ["dipole_56.00m", "zen_counts2mv"],
                ),
                ("measurement_azimuth", 90.0),
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
                ("sample_rate", 256.0),
                ("time_period.end", "2022-05-17T15:54:42+00:00"),
                ("time_period.start", "2022-05-17T13:09:58+00:00"),
                ("type", "electric"),
                ("units", "digital counts"),
            ]
        )

        for key, value in ey.items():
            with self.subTest(key):
                self.assertEqual(
                    value, self.z3d.channel_metadata.get_attr_from_name(key)
                )

    def test_run_metadata(self):
        rm = OrderedDict(
            [
                ("acquired_by.author", ""),
                ("channels_recorded_auxiliary", []),
                ("channels_recorded_electric", []),
                ("channels_recorded_magnetic", []),
                ("data_logger.firmware.author", None),
                ("data_logger.firmware.name", None),
                ("data_logger.firmware.version", "4147.0"),
                ("data_logger.id", "ZEN024"),
                ("data_logger.manufacturer", "Zonge International"),
                ("data_logger.model", "ZEN"),
                ("data_logger.timing_system.drift", 0.0),
                ("data_logger.timing_system.type", "GPS"),
                ("data_logger.timing_system.uncertainty", 0.0),
                ("data_logger.type", None),
                ("data_type", "MTBB"),
                ("id", "sr256_001"),
                ("sample_rate", 256.0),
                ("time_period.end", "2022-05-17T15:54:42+00:00"),
                ("time_period.start", "2022-05-17T13:09:58+00:00"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                self.assertEqual(
                    value, self.z3d.run_metadata.get_attr_from_name(key)
                )

    def test_station_metadata(self):
        sm = OrderedDict(
            [
                ("acquired_by.name", ""),
                ("channels_recorded", []),
                ("data_type", "BBMT"),
                ("fdsn.id", "100"),
                ("geographic_name", None),
                ("id", "100"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 0.0),
                ("location.elevation", 1456.3),
                ("location.latitude", 40.49757833327694),
                ("location.longitude", -116.8211900230401),
                ("orientation.method", None),
                ("orientation.reference_frame", "geographic"),
                ("provenance.archive.name", None),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.creator.name", None),
                ("provenance.software.author", None),
                ("provenance.software.name", None),
                ("provenance.software.version", None),
                ("provenance.submitter.email", None),
                ("provenance.submitter.name", None),
                ("provenance.submitter.organization", None),
                ("release_license", "CC0-1.0"),
                ("run_list", []),
                ("time_period.end", "2022-05-17T15:54:42+00:00"),
                ("time_period.start", "2022-05-17T13:09:58+00:00"),
            ]
        )

        for key, value in sm.items():
            with self.subTest(key):
                self.assertEqual(
                    value, self.z3d.station_metadata.get_attr_from_name(key)
                )

    def test_filters(self):
        # zr = FrequencyResponseTableFilter(
        #     **OrderedDict(
        #         [
        #             ("amplitudes", np.array([1.00153])),
        #             ("calibration_date", "1980-01-01"),
        #             ("comments", "data logger response read from z3d file"),
        #             ("frequencies", np.array([2.0])),
        #             ("gain", 1.0),
        #             ("instrument_type", None),
        #             ("name", "zen024_256_response"),
        #             ("phases", np.array([-1.5333299999999999])),
        #             ("type", "frequency response table"),
        #             ("units_in", "mV"),
        #             ("units_out", "mV"),
        #         ]
        #     )
        # )

        df = CoefficientFilter(
            **OrderedDict(
                [
                    ("calibration_date", "1980-01-01"),
                    ("comments", "convert to electric field"),
                    ("gain", 0.056),
                    ("name", "dipole_56.00m"),
                    ("type", "coefficient"),
                    ("units_out", "mV"),
                    ("units_in", "mV/km"),
                ]
            )
        )

        cf = CoefficientFilter(
            **OrderedDict(
                [
                    ("calibration_date", "1980-01-01"),
                    ("comments", "digital counts to millivolts"),
                    ("gain", 1048576000.000055),
                    ("name", "zen_counts2mv"),
                    ("type", "coefficient"),
                    ("units_out", "count"),
                    ("units_in", "mV"),
                ]
            )
        )

        # with self.subTest("test zen response"):
        #     self.assertEqual(None, self.z3d.zen_response)
        #     # self.assertDictEqual(
        #     #     self.z3d.zen_response.to_dict(single=True),
        #     #     zr.to_dict(single=True),
        #     # )
        with self.subTest("test_dipole_filter"):

            self.assertDictEqual(
                self.z3d.dipole_filter.to_dict(single=True),
                df.to_dict(single=True),
            )

        with self.subTest("test_conversion_filter"):

            self.assertEqual(
                self.z3d.counts2mv_filter.to_dict(single=True),
                cf.to_dict(single=True),
            )

        with self.subTest("channel_response"):

            cr = ChannelResponseFilter(filters_list=[df, cf])
            self.assertListEqual(
                cr.filters_list, self.z3d.channel_response.filters_list
            )


@unittest.skipIf(
    "peacock" not in str(Path(__file__).as_posix()), "local files"
)
class TestZ3DHY(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fn = Path(
            r"c:\Users\jpeacock\OneDrive - DOI\mt\example_z3d_data\bm100_20220517_131017_256_HY.Z3D"
        )
        self.z3d = Z3D(self.fn)
        self.z3d.read_z3d()

        self.maxDiff = None

        self.cr = FrequencyResponseTableFilter(
            **OrderedDict(
                [
                    (
                        "amplitudes",
                        np.array(
                            [
                                0.27649,
                                0.39498,
                                0.59248,
                                0.75047,
                                1.14545,
                                1.54044,
                                2.3304,
                                3.08087,
                                4.62131,
                                6.16174,
                                9.25667,
                                12.3343,
                                18.2536,
                                24.0094,
                                35.0682,
                                44.3682,
                                59.9492,
                                70.4913,
                                83.2834,
                                89.5798,
                                95.028,
                                97.1974,
                                98.8414,
                                99.5336,
                                99.975,
                                100.138,
                                100.246,
                                100.26,
                                100.332,
                                100.33,
                                100.37,
                                100.355,
                                100.414,
                                100.275,
                                100.427,
                                100.382,
                                100.307,
                                100.353,
                                99.8983,
                                99.0787,
                                90.2819,
                                68.2665,
                                27.3347,
                                11.8179,
                                3.70505,
                                1.62308,
                                0.5053,
                                0.24206,
                            ]
                        ),
                    ),
                    ("calibration_date", "1980-01-01"),
                    ("comments", "induction coil response read from z3d file"),
                    (
                        "frequencies",
                        np.array(
                            [
                                [
                                    1.16568582e-04,
                                    1.55424829e-04,
                                    2.33136527e-04,
                                    3.10850294e-04,
                                    4.66274645e-04,
                                    6.21698996e-04,
                                    9.32549290e-04,
                                    1.24339799e-03,
                                    1.86510495e-03,
                                    2.48679599e-03,
                                    3.73019398e-03,
                                    4.97359197e-03,
                                    7.46038796e-03,
                                    9.94718394e-03,
                                    1.49207759e-02,
                                    1.98943679e-02,
                                    2.98415518e-02,
                                    3.97887358e-02,
                                    5.96831037e-02,
                                    7.95774715e-02,
                                    1.19366207e-01,
                                    1.59154943e-01,
                                    2.38732415e-01,
                                    3.18309886e-01,
                                    4.77464829e-01,
                                    6.36619772e-01,
                                    9.54929659e-01,
                                    1.27323954e00,
                                    1.90985932e00,
                                    2.54647909e00,
                                    3.81971863e00,
                                    5.09295818e00,
                                    7.63943727e00,
                                    1.01859164e01,
                                    1.52788745e01,
                                    2.03718327e01,
                                    3.05577491e01,
                                    4.07436654e01,
                                    6.11154981e01,
                                    8.14873309e01,
                                    1.22230996e02,
                                    1.62974662e02,
                                    2.44461993e02,
                                    3.25949323e02,
                                    4.88923985e02,
                                    6.51898647e02,
                                    9.77847970e02,
                                    1.30379729e03,
                                ]
                            ]
                        ),
                    ),
                    ("gain", 1.0),
                    ("instrument_type", None),
                    ("name", "ant4_2324_response"),
                    (
                        "phases",
                        np.array(
                            [
                                1.5702e00,
                                1.5628e00,
                                1.5545e00,
                                1.5496e00,
                                1.5409e00,
                                1.5348e00,
                                1.5262e00,
                                1.5205e00,
                                1.5121e00,
                                1.5062e00,
                                1.4792e00,
                                1.4470e00,
                                1.3855e00,
                                1.3300e00,
                                1.2169e00,
                                1.1118e00,
                                9.2770e-01,
                                7.9150e-01,
                                5.9040e-01,
                                4.6760e-01,
                                3.2250e-01,
                                2.4570e-01,
                                1.6350e-01,
                                1.2190e-01,
                                7.8400e-02,
                                5.5100e-02,
                                3.0300e-02,
                                1.5700e-02,
                                -2.9000e-03,
                                -1.6100e-02,
                                -3.7800e-02,
                                -5.6100e-02,
                                -9.1000e-02,
                                -1.2490e-01,
                                -1.9050e-01,
                                -2.5650e-01,
                                -3.8930e-01,
                                -5.2140e-01,
                                -7.9710e-01,
                                -1.0885e00,
                                -1.7401e00,
                                -2.4110e00,
                                2.9802e00,
                                2.5876e00,
                                2.1997e00,
                                1.9765e00,
                                1.7768e00,
                                1.5969e00,
                            ]
                        ),
                    ),
                    ("type", "frequency response table"),
                    ("units_in", "mV"),
                    ("units_out", "nT"),
                ]
            )
        )

        self.zr = FrequencyResponseTableFilter(
            **OrderedDict(
                [
                    ("amplitudes", np.array([0.999878])),
                    ("calibration_date", "1980-01-01"),
                    ("comments", "data logger response read from z3d file"),
                    ("frequencies", np.array([2.0])),
                    ("gain", 1.0),
                    ("instrument_type", None),
                    ("name", "zen024_256_response"),
                    ("phases", np.array([-1.53334])),
                    ("type", "frequency response table"),
                    ("units_in", "mV"),
                    ("units_out", "mV"),
                ]
            )
        )

        self.cf = CoefficientFilter(
            **OrderedDict(
                [
                    ("calibration_date", "1980-01-01"),
                    ("comments", "digital counts to millivolts"),
                    ("gain", 1048576000.000055),
                    ("name", "zen_counts2mv"),
                    ("type", "coefficient"),
                    ("units_out", "count"),
                    ("units_in", "mV"),
                ]
            )
        )

    def test_channel_metadata(self):
        ey = OrderedDict(
            [
                ("channel_number", 2),
                ("component", "hy"),
                ("data_quality.rating.value", 0),
                ("filter.applied", [False, False]),
                (
                    "filter.name",
                    [
                        "ant4_2324_response",
                        "zen_counts2mv",
                    ],
                ),
                ("h_field_max.end", 0.02879215431213228),
                ("h_field_max.start", 0.03145987892150714),
                ("h_field_min.end", 0.02772834396362159),
                ("h_field_min.start", 0.02886334419250337),
                ("location.elevation", 0.0),
                ("location.latitude", 0.0),
                ("location.longitude", 0.0),
                ("measurement_azimuth", 90.0),
                ("measurement_tilt", 0.0),
                ("sample_rate", 256.0),
                ("sensor.id", ["2324"]),
                ("sensor.manufacturer", ["Geotell"]),
                ("sensor.model", ["ANT-4"]),
                ("sensor.type", ["induction coil"]),
                ("time_period.end", "2022-05-17T15:54:42+00:00"),
                ("time_period.start", "2022-05-17T13:09:58+00:00"),
                ("type", "magnetic"),
                ("units", "digital counts"),
            ]
        )

        for key, value in ey.items():
            with self.subTest(key):
                self.assertEqual(
                    value, self.z3d.channel_metadata.get_attr_from_name(key)
                )

    def test_run_metadata(self):
        rm = OrderedDict(
            [
                ("acquired_by.author", ""),
                ("channels_recorded_auxiliary", []),
                ("channels_recorded_electric", []),
                ("channels_recorded_magnetic", []),
                ("data_logger.firmware.author", None),
                ("data_logger.firmware.name", None),
                ("data_logger.firmware.version", "4147.0"),
                ("data_logger.id", "ZEN024"),
                ("data_logger.manufacturer", "Zonge International"),
                ("data_logger.model", "ZEN"),
                ("data_logger.timing_system.drift", 0.0),
                ("data_logger.timing_system.type", "GPS"),
                ("data_logger.timing_system.uncertainty", 0.0),
                ("data_logger.type", None),
                ("data_type", "MTBB"),
                ("id", "sr256_001"),
                ("sample_rate", 256.0),
                ("time_period.end", "2022-05-17T15:54:42+00:00"),
                ("time_period.start", "2022-05-17T13:09:58+00:00"),
            ]
        )

        for key, value in rm.items():
            with self.subTest(key):
                self.assertEqual(
                    value, self.z3d.run_metadata.get_attr_from_name(key)
                )

    def test_station_metadata(self):
        sm = OrderedDict(
            [
                ("acquired_by.name", ""),
                ("channels_recorded", []),
                ("data_type", "BBMT"),
                ("fdsn.id", "100"),
                ("geographic_name", None),
                ("id", "100"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 0.0),
                ("location.elevation", 1456.3),
                ("location.latitude", 40.49757833327694),
                ("location.longitude", -116.8211900230401),
                ("orientation.method", None),
                ("orientation.reference_frame", "geographic"),
                ("provenance.archive.name", None),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.creator.name", None),
                ("provenance.software.author", None),
                ("provenance.software.name", None),
                ("provenance.software.version", None),
                ("provenance.submitter.email", None),
                ("provenance.submitter.name", None),
                ("provenance.submitter.organization", None),
                ("release_license", "CC0-1.0"),
                ("run_list", []),
                ("time_period.end", "2022-05-17T15:54:42+00:00"),
                ("time_period.start", "2022-05-17T13:09:58+00:00"),
            ]
        )

        for key, value in sm.items():
            with self.subTest(key):
                self.assertEqual(
                    value, self.z3d.station_metadata.get_attr_from_name(key)
                )

    def test_zen_esponse(self):
        self.assertEqual(None, self.z3d.zen_response)
        # self.assertDictEqual(
        #     self.z3d.zen_response.to_dict(single=True),
        #     self.zr.to_dict(single=True),
        # )

    def test_coil_response_filter(self):

        with self.subTest("frequency"):
            self.assertTrue(
                np.isclose(
                    self.z3d.coil_response.frequencies, self.cr.frequencies
                ).all()
            )

        with self.subTest("amplitude"):
            self.assertTrue(
                np.isclose(
                    self.z3d.coil_response.amplitudes, self.cr.amplitudes
                ).all()
            )

        with self.subTest("phase"):
            self.assertTrue(
                np.isclose(self.z3d.coil_response.phases, self.cr.phases).all()
            )

    def test_conversion_filter(self):

        self.assertEqual(
            self.z3d.counts2mv_filter.to_dict(single=True),
            self.cf.to_dict(single=True),
        )

    def channel_response(self):

        cr = ChannelResponseFilter(filters_list=[self.cf, self.cr])

        self.assertListEqual(
            cr.filters_list, self.z3d.channel_response.filters_list
        )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
