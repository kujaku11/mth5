# -*- coding: utf-8 -*-
"""
Test reading lemi files


:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT
"""

# ==============================================================================
# Imports
# ==============================================================================
import unittest
from collections import OrderedDict
from pathlib import Path

from mth5.io.lemi import LEMI424
from mt_metadata.timeseries import Station, Run

# ==============================================================================
# make an example text string similar to lemi data.
lemi_str = (
    "2020 10 04 00 00 00 23772.512   238.148 41845.187  33.88  25.87   142.134   -45.060   213.787     8.224 12.78 2199.0 3404.83963 N 10712.84474 W 12 2 0 \n"
    "2020 10 04 00 00 01 23772.514   238.159 41845.202  33.89  25.87   142.014   -45.172   213.662     8.112 12.78 2199.0 3404.83965 N 10712.84471 W 12 2 0 \n"
    "2020 10 04 00 00 02 23772.522   238.149 41845.191  33.90  25.87   142.033   -45.156   213.688     8.136 12.78 2199.1 3404.83964 N 10712.84469 W 12 2 0 \n"
    "2020 10 04 00 00 03 23772.537   238.143 41845.210  33.89  25.87   142.023   -45.122   213.679     8.177 12.78 2199.2 3404.83965 N 10712.84467 W 12 2 0 \n"
    "2020 10 04 00 00 04 23772.563   238.150 41845.198  33.88  25.87   141.961   -45.128   213.621     8.165 12.78 2199.2 3404.83966 N 10712.84465 W 12 2 0 \n"
    "2020 10 04 00 00 05 23772.563   238.168 41845.195  33.89  25.87   141.898   -45.200   213.554     8.104 12.78 2199.2 3404.83968 N 10712.84463 W 12 2 0 \n"
    "2020 10 04 00 00 06 23772.582   238.156 41845.198  33.88  25.87   141.894   -45.161   213.559     8.142 12.78 2199.2 3404.83968 N 10712.84461 W 12 2 0 \n"
    "2020 10 04 00 00 07 23772.574   238.157 41845.187  33.89  25.87   141.842   -45.183   213.509     8.120 12.78 2199.2 3404.83968 N 10712.84460 W 12 2 0 \n"
    "2020 10 04 00 00 08 23772.588   238.180 41845.187  33.88  25.87   141.766   -45.227   213.435     8.057 12.78 2199.1 3404.83967 N 10712.84458 W 12 2 0 \n"
    "2020 10 04 00 00 09 23772.586   238.171 41845.163  33.90  25.87   141.702   -45.240   213.374     8.044 12.78 2199.1 3404.83964 N 10712.84454 W 12 2 0 \n"
    "2020 10 04 00 00 10 23772.594   238.181 41845.179  33.91  25.87   141.699   -45.261   213.377     8.025 12.78 2199.1 3404.83962 N 10712.84449 W 12 2 0 \n"
    "2020 10 04 00 00 11 23772.592   238.211 41845.187  33.85  25.87   141.617   -45.319   213.295     7.953 12.78 2199.1 3404.83961 N 10712.84445 W 12 2 0 \n"
    "2020 10 04 00 00 12 23772.588   238.196 41845.183  33.92  25.87   141.542   -45.349   213.225     7.921 12.78 2199.1 3404.83959 N 10712.84440 W 12 2 0 \n"
    "2020 10 04 00 00 13 23772.594   238.207 41845.167  33.88  25.87   141.485   -45.369   213.172     7.899 12.78 2199.1 3404.83956 N 10712.84435 W 12 2 0 \n"
    "2020 10 04 00 00 14 23772.592   238.198 41845.175  33.88  25.87   141.427   -45.452   213.119     7.812 12.78 2199.1 3404.83953 N 10712.84429 W 12 2 0 \n"
    "2020 10 04 00 00 15 23772.596   238.193 41845.191  33.88  25.87   141.361   -45.505   213.058     7.761 12.78 2199.1 3404.83951 N 10712.84424 W 12 2 0 \n"
    "2020 10 04 00 00 16 23772.592   238.207 41845.202  33.90  25.87   141.334   -45.544   213.036     7.729 12.78 2199.1 3404.83950 N 10712.84420 W 12 2 0 \n"
    "2020 10 04 00 00 17 23772.578   238.233 41845.198  33.91  25.87   141.340   -45.566   213.052     7.697 12.78 2199.1 3404.83948 N 10712.84416 W 12 2 0 \n"
    "2020 10 04 00 00 18 23772.572   238.245 41845.202  33.88  25.87   141.214   -45.636   212.936     7.625 12.78 2199.0 3404.83946 N 10712.84412 W 12 2 0 \n"
    "2020 10 04 00 00 19 23772.549   238.260 41845.195  33.88  25.87   141.211   -45.659   212.942     7.610 12.78 2198.9 3404.83945 N 10712.84407 W 12 2 0 \n"
    "2020 10 04 00 00 20 23772.555   238.261 41845.187  33.88  25.87   141.208   -45.658   212.950     7.600 12.78 2198.9 3404.83945 N 10712.84403 W 12 2 0 \n"
    "2020 10 04 00 00 21 23772.545   238.265 41845.210  33.86  25.87   141.147   -45.711   212.890     7.560 12.78 2198.9 3404.83944 N 10712.84400 W 12 2 0 \n"
    "2020 10 04 00 00 22 23772.535   238.272 41845.191  33.90  25.87   141.164   -45.727   212.917     7.552 12.78 2198.8 3404.83944 N 10712.84398 W 12 2 0 \n"
    "2020 10 04 00 00 23 23772.525   238.286 41845.210  33.91  25.87   141.178   -45.707   212.939     7.555 12.78 2198.8 3404.83943 N 10712.84395 W 12 2 0 \n"
    "2020 10 04 00 00 24 23772.512   238.275 41845.214  33.88  25.87   141.145   -45.751   212.917     7.510 12.78 2198.7 3404.83943 N 10712.84391 W 12 2 0 \n"
    "2020 10 04 00 00 25 23772.482   238.264 41845.187  33.89  25.87   141.158   -45.717   212.937     7.548 12.78 2198.6 3404.83944 N 10712.84387 W 12 2 0 \n"
    "2020 10 04 00 00 26 23772.463   238.300 41845.206  33.88  25.87   141.150   -45.732   212.941     7.544 12.78 2198.6 3404.83944 N 10712.84385 W 12 2 0 \n"
    "2020 10 04 00 00 27 23772.457   238.317 41845.202  33.92  25.87   141.145   -45.777   212.942     7.515 12.78 2198.5 3404.83944 N 10712.84382 W 12 2 0 \n"
    "2020 10 04 00 00 28 23772.441   238.300 41845.206  33.87  25.87   141.160   -45.729   212.961     7.556 12.78 2198.5 3404.83944 N 10712.84379 W 12 2 0 \n"
    "2020 10 04 00 00 29 23772.434   238.292 41845.210  33.85  25.87   141.156   -45.732   212.962     7.548 12.78 2198.5 3404.83943 N 10712.84377 W 12 2 0 \n"
    "2020 10 04 00 00 30 23772.416   238.302 41845.187  33.86  25.87   141.139   -45.777   212.942     7.522 12.78 2198.4 3404.83942 N 10712.84376 W 12 2 0 \n"
    "2020 10 04 00 00 31 23772.408   238.305 41845.226  33.88  25.87   141.187   -45.739   212.997     7.566 12.78 2198.4 3404.83941 N 10712.84376 W 12 2 0 \n"
    "2020 10 04 00 00 32 23772.383   238.302 41845.214  33.90  25.87   141.212   -45.727   213.019     7.578 12.78 2198.3 3404.83940 N 10712.84375 W 12 2 0 \n"
    "2020 10 04 00 00 33 23772.363   238.303 41845.222  33.88  25.87   141.166   -45.808   212.967     7.514 12.78 2198.4 3404.83940 N 10712.84374 W 12 2 0 \n"
    "2020 10 04 00 00 34 23772.348   238.291 41845.195  33.90  25.87   141.251   -45.752   213.052     7.574 12.78 2198.3 3404.83939 N 10712.84374 W 12 2 0 \n"
    "2020 10 04 00 00 35 23772.352   238.289 41845.198  33.89  25.87   141.303   -45.774   213.101     7.561 12.78 2198.3 3404.83939 N 10712.84374 W 12 2 0 \n"
    "2020 10 04 00 00 36 23772.356   238.310 41845.226  33.91  25.87   141.295   -45.740   213.092     7.574 12.78 2198.3 3404.83937 N 10712.84373 W 12 2 0 \n"
    "2020 10 04 00 00 37 23772.309   238.290 41845.206  33.84  25.87   141.370   -45.734   213.159     7.587 12.78 2198.3 3404.83936 N 10712.84373 W 12 2 0 \n"
    "2020 10 04 00 00 38 23772.275   238.296 41845.187  33.88  25.87   141.404   -45.714   213.194     7.612 12.78 2198.3 3404.83935 N 10712.84372 W 12 2 0 \n"
    "2020 10 04 00 00 39 23772.287   238.303 41845.198  33.88  25.87   141.434   -45.702   213.214     7.639 12.78 2198.3 3404.83935 N 10712.84371 W 12 2 0 \n"
    "2020 10 04 00 00 40 23772.281   238.288 41845.214  33.88  25.87   141.502   -45.688   213.279     7.666 12.78 2198.3 3404.83934 N 10712.84371 W 12 2 0 \n"
    "2020 10 04 00 00 41 23772.258   238.261 41845.198  33.89  25.87   141.571   -45.650   213.340     7.709 12.78 2198.3 3404.83932 N 10712.84371 W 12 2 0 \n"
    "2020 10 04 00 00 42 23772.254   238.280 41845.187  33.87  25.87   141.631   -45.636   213.390     7.733 12.78 2198.3 3404.83930 N 10712.84370 W 12 2 0 \n"
    "2020 10 04 00 00 43 23772.250   238.271 41845.191  33.88  25.87   141.705   -45.577   213.457     7.804 12.78 2198.3 3404.83928 N 10712.84371 W 12 2 0 \n"
    "2020 10 04 00 00 44 23772.223   238.255 41845.183  33.88  25.87   141.771   -45.549   213.517     7.832 12.78 2198.3 3404.83926 N 10712.84371 W 12 2 0 \n"
    "2020 10 04 00 00 45 23772.238   238.248 41845.202  33.84  25.87   141.824   -45.558   213.558     7.866 12.78 2198.4 3404.83924 N 10712.84372 W 12 2 0 \n"
    "2020 10 04 00 00 46 23772.252   238.244 41845.187  33.87  25.87   141.921   -45.475   213.649     7.964 12.78 2198.4 3404.83921 N 10712.84372 W 12 2 0 \n"
    "2020 10 04 00 00 47 23772.219   238.250 41845.159  33.88  25.87   141.968   -45.422   213.686     8.003 12.78 2198.5 3404.83918 N 10712.84371 W 12 2 0 \n"
    "2020 10 04 00 00 48 23772.221   238.231 41845.175  33.88  25.87   142.030   -45.401   213.737     8.024 12.78 2198.5 3404.83914 N 10712.84371 W 12 2 0 \n"
    "2020 10 04 00 00 49 23772.223   238.210 41845.179  33.89  25.87   142.053   -45.315   213.754     8.113 12.78 2198.6 3404.83911 N 10712.84370 W 12 2 0 \n"
    "2020 10 04 00 00 50 23772.242   238.208 41845.183  33.88  25.87   142.134   -45.263   213.824     8.160 12.78 2198.6 3404.83909 N 10712.84370 W 12 2 0 \n"
    "2020 10 04 00 00 51 23772.266   238.214 41845.179  33.88  25.87   142.133   -45.271   213.814     8.153 12.78 2198.6 3404.83908 N 10712.84371 W 12 2 0 \n"
    "2020 10 04 00 00 52 23772.256   238.217 41845.179  33.88  25.87   142.153   -45.261   213.854     8.172 12.78 2198.6 3404.83907 N 10712.84372 W 12 2 0 \n"
    "2020 10 04 00 00 53 23772.268   238.199 41845.179  33.88  25.87   142.201   -45.190   213.900     8.231 12.78 2198.6 3404.83906 N 10712.84372 W 12 2 0 \n"
    "2020 10 04 00 00 54 23772.281   238.214 41845.159  33.88  25.87   142.212   -45.187   213.881     8.238 12.78 2198.6 3404.83904 N 10712.84373 W 12 2 0 \n"
    "2020 10 04 00 00 55 23772.279   238.207 41845.187  33.87  25.87   142.184   -45.185   213.846     8.239 12.78 2198.6 3404.83904 N 10712.84375 W 12 2 0 \n"
    "2020 10 04 00 00 56 23772.287   238.175 41845.175  33.87  25.87   142.252   -45.134   213.911     8.285 12.78 2198.6 3404.83903 N 10712.84377 W 12 2 0 \n"
    "2020 10 04 00 00 57 23772.301   238.183 41845.183  33.85  25.87   142.206   -45.166   213.860     8.254 12.78 2198.6 3404.83902 N 10712.84378 W 12 2 0 \n"
    "2020 10 04 00 00 58 23772.297   238.175 41845.183  33.89  25.87   142.161   -45.196   213.810     8.219 12.78 2198.5 3404.83901 N 10712.84379 W 12 2 0 \n"
    "2020 10 04 00 00 59 23772.316   238.177 41845.183  33.88  25.87   142.233   -45.164   213.878     8.242 12.78 2198.6 3404.83902 N 10712.84380 W 12 2 0 "
)

# write to a file
cwd = Path().cwd()
lemi_fn = cwd.joinpath("example_lemi.txt")
with open(lemi_fn, "w") as fid:
    fid.write(lemi_str)

# ==============================================================================


class TestLEMI424(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.maxDiff = None
        self.lemi_obj = LEMI424(lemi_fn)
        self.lemi_obj.read()

        self.station_metadata = Station()
        self.station_metadata.from_dict(
            OrderedDict(
                [
                    ("acquired_by.name", None),
                    (
                        "channels_recorded",
                        [
                            "bx",
                            "by",
                            "bz",
                            "e1",
                            "e2",
                            "temperature_e",
                            "temperature_h",
                        ],
                    ),
                    ("data_type", "BBMT"),
                    ("geographic_name", None),
                    ("id", None),
                    ("location.declination.model", "WMM"),
                    ("location.declination.value", 0.0),
                    ("location.elevation", 2198.6),
                    ("location.latitude", 34.080657083333335),
                    ("location.longitude", -107.21406316666668),
                    ("orientation.method", None),
                    ("orientation.reference_frame", "geographic"),
                    ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                    ("provenance.software.author", "none"),
                    ("provenance.software.name", None),
                    ("provenance.software.version", None),
                    ("provenance.submitter.email", None),
                    ("provenance.submitter.organization", None),
                    ("run_list", ["a"]),
                    ("time_period.end", "2020-10-04T00:00:59+00:00"),
                    ("time_period.start", "2020-10-04T00:00:00+00:00"),
                ]
            )
        )

        self.station_metadata.provenance.creation_time = (
            self.lemi_obj.station_metadata.provenance.creation_time
        )

        self.run_metadata = Run()
        self.run_metadata.from_dict(
            OrderedDict(
                [
                    (
                        "channels_recorded_auxiliary",
                        ["temperature_e", "temperature_h"],
                    ),
                    ("channels_recorded_electric", ["e1", "e2"]),
                    ("channels_recorded_magnetic", ["bx", "by", "bz"]),
                    ("data_logger.firmware.author", None),
                    ("data_logger.firmware.name", None),
                    ("data_logger.firmware.version", None),
                    ("data_logger.id", None),
                    ("data_logger.manufacturer", "LEMI"),
                    ("data_logger.model", "LEMI424"),
                    ("data_logger.power_source.voltage.end", 12.78),
                    ("data_logger.power_source.voltage.start", 12.78),
                    ("data_logger.timing_system.drift", 0.0),
                    ("data_logger.timing_system.type", "GPS"),
                    ("data_logger.timing_system.uncertainty", 0.0),
                    ("data_logger.type", None),
                    ("data_type", "BBMT"),
                    ("id", "a"),
                    ("sample_rate", 1.0),
                    ("time_period.end", "2020-10-04T00:00:59+00:00"),
                    ("time_period.start", "2020-10-04T00:00:00+00:00"),
                ]
            )
        )
        self.station_metadata.add_run(self.run_metadata)

    def test_has_data(self):
        self.assertTrue(self.lemi_obj._has_data())

    def test_start(self):
        self.assertEqual("2020-10-04T00:00:00+00:00", self.lemi_obj.start.isoformat())

    def test_end(self):
        self.assertEqual("2020-10-04T00:00:59+00:00", self.lemi_obj.end.isoformat())

    def test_n_samples(self):
        self.assertEqual(60, self.lemi_obj.n_samples)

    def test_latitude(self):
        self.assertAlmostEqual(34.080657083333335, self.lemi_obj.latitude, 5)

    def test_longitude(self):
        self.assertAlmostEqual(-107.21406316666668, self.lemi_obj.longitude, 5)

    def test_elevation(self):
        self.assertAlmostEqual(2198.6, self.lemi_obj.elevation, 2)

    def test_df_columns(self):
        self.assertListEqual(
            self.lemi_obj.data_column_names[1:],
            self.lemi_obj.data.columns.to_list(),
        )

    def test_station_metadata(self):
        self.station_metadata.provenance.creation_time = (
            self.lemi_obj.station_metadata.provenance.creation_time
        )
        sd = self.station_metadata.to_dict(single=True)
        ld = self.lemi_obj.station_metadata.to_dict(single=True)
        try:
            sd.pop("provenance.creation_time")
            ld.pop("provenance.creation_time")
        except KeyError:
            pass
        self.assertDictEqual(sd, ld)

    def test_run_metadata(self):
        self.assertDictEqual(
            self.run_metadata.to_dict(single=True),
            self.lemi_obj.run_metadata.to_dict(single=True),
        )

    def test_to_runts(self):
        r = self.lemi_obj.to_run_ts()
        self.station_metadata.provenance.creation_time = (
            r.station_metadata.provenance.creation_time
        )

        with self.subTest("station_metadata"):
            dict1 = self.station_metadata.to_dict(single=True)
            dict2 = self.lemi_obj.station_metadata.to_dict(single=True)

            # remove the creation time from the dicts for comparison
            dict1.pop("provenance.creation_time", None)
            dict2.pop("provenance.creation_time", None)
            # compare the two dictionaries
            self.assertDictEqual(
                dict1,
                dict2,
            )
            # self.assertDictEqual(
            #     self.station_metadata.to_dict(single=True),
            #     r.station_metadata.to_dict(single=True),
            # )

        with self.subTest("run_metadata"):
            self.assertDictEqual(
                self.run_metadata.to_dict(single=True),
                r.run_metadata.to_dict(single=True),
            )

        with self.subTest("channels"):
            self.assertListEqual(
                [
                    "bx",
                    "by",
                    "bz",
                    "e1",
                    "e2",
                    "temperature_e",
                    "temperature_h",
                ],
                r.channels,
            )

        with self.subTest("n samples"):
            self.assertEqual(60, r.dataset.dims["time"])


class TestLEMI424Metadata(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.lemi_obj = LEMI424(lemi_fn)
        self.lemi_obj.read_metadata()

    def test_has_data(self):
        self.assertTrue(self.lemi_obj._has_data())

    def test_start(self):
        self.assertEqual("2020-10-04T00:00:00+00:00", self.lemi_obj.start.isoformat())

    def test_end(self):
        self.assertEqual("2020-10-04T00:00:59+00:00", self.lemi_obj.end.isoformat())

    def test_n_samples(self):
        self.assertEqual(2, self.lemi_obj.n_samples)

    def test_latitude(self):
        self.assertAlmostEqual(34.080655416666666, self.lemi_obj.latitude, 5)

    def test_longitude(self):
        self.assertAlmostEqual(-107.21407116666666, self.lemi_obj.longitude, 5)

    def test_elevation(self):
        self.assertAlmostEqual(2198.8, self.lemi_obj.elevation, 2)

    def test_df_columns(self):
        self.assertListEqual(
            self.lemi_obj.data_column_names[1:],
            self.lemi_obj.data.columns.to_list(),
        )


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
