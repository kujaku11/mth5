# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:24:28 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import unittest
from pathlib import Path
from collections import OrderedDict

from mth5.mth5 import MTH5

from mt_metadata.transfer_functions.core import TF
from mt_metadata import TF_XML


fn_path = Path(__file__).parent
# =============================================================================


class TestTFGroup(unittest.TestCase):
    @classmethod
    def setUpClass(self):

        self.maxDiff = None
        self.fn = fn_path.joinpath("test.mth5")
        self.mth5_obj = MTH5(file_version="0.2.0")
        self.mth5_obj.open_mth5(self.fn, mode="a")

        self.tf_obj = TF(TF_XML)

        self.tf_group = self.mth5_obj.add_transfer_function(self.tf_obj)
        self.tf_h5 = self.mth5_obj.get_transfer_function(
            self.tf_obj.station,
            self.tf_obj.tf_id,
            survey=self.tf_obj.survey_metadata.id,
        )

    def test_survey_metadata(self):
        meta_dict = OrderedDict(
            [
                ("acquired_by.author", "National Geoelectromagnetic Facility"),
                (
                    "citation_dataset.doi",
                    "doi:10.17611/DP/EMTF/USMTARRAY/SOUTH",
                ),
                (
                    "citation_dataset.title",
                    "USMTArray South Magnetotelluric Transfer Functions",
                ),
                ("citation_dataset.year", "2020-2023"),
                ("citation_journal.doi", None),
                (
                    "comments",
                    "The USMTArray-CONUS South campaign was carried out through a cooperative agreement between\nthe U.S. Geological Survey (USGS) and Oregon State University (OSU). A subset of 40 stations\nin the SW US were funded through NASA grant 80NSSC19K0232.\nLand permitting, data acquisition, quality control and field processing were\ncarried out by Green Geophysics with project management and instrument/engineering\nsupport from OSU and Chaytus Engineering, respectively.\nProgram oversight, definitive data processing and data archiving were provided\nby the USGS Geomagnetism Program and the Geology, Geophysics and Geochemistry Science Centers.\nWe thank the U.S. Forest Service, the Bureau of Land Management, the National Park Service,\nthe Department of Defense, numerous state land offices and the many private landowners\nwho permitted land access to acquire the USMTArray data.",
                ),
                ("country", "USA"),
                ("datum", "WGS84"),
                ("geographic_name", "CONUS South"),
                ("id", "CONUS South"),
                ("name", None),
                ("northwest_corner.latitude", 0.0),
                ("northwest_corner.longitude", 0.0),
                ("project", "USMTArray"),
                ("project_lead.email", None),
                ("project_lead.organization", None),
                ("release_license", "CC-0"),
                ("southeast_corner.latitude", 0.0),
                ("southeast_corner.longitude", 0.0),
                ("summary", "Magnetotelluric Transfer Functions"),
                ("time_period.end_date", "2020-10-07"),
                ("time_period.start_date", "2020-09-20"),
            ]
        )

        h5_meta_dict = self.tf_h5.survey_metadata.to_dict(single=True)
        h5_meta_dict.pop("mth5_type")
        h5_meta_dict.pop("hdf5_reference")

        self.assertDictEqual(meta_dict, h5_meta_dict)

    def test_station_metadta(self):

        meta_dict = OrderedDict(
            [
                ("acquired_by.author", "National Geoelectromagnetic Facility"),
                ("channels_recorded", ["ex", "ey", "hx", "hy", "hz"]),
                ("data_type", "mt"),
                ("fdsn.id", "USMTArray.NMX20.2020"),
                ("geographic_name", "Nations Draw, NM, USA"),
                ("id", "NMX20"),
                ("location.datum", "WGS84"),
                ("location.declination.epoch", "2020.0"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 9.0899999999999999),
                ("location.elevation", 1940.05),
                ("location.latitude", 34.470528),
                ("location.longitude", -108.712288),
                ("orientation.angle_to_geographic_north", 0.0),
                ("orientation.method", None),
                ("orientation.reference_frame", "geographic"),
                ("provenance.creation_time", "2021-03-17T14:47:44+00:00"),
                ("provenance.software.author", "none"),
                (
                    "provenance.software.name",
                    "EMTF File Conversion Utilities 4.0",
                ),
                ("provenance.software.version", None),
                ("provenance.submitter.author", "Anna Kelbert"),
                ("provenance.submitter.email", "akelbert@usgs.gov"),
                (
                    "provenance.submitter.organization",
                    "U.S. Geological Survey, Geomagnetism Program",
                ),
                ("run_list", ["NMX20a", "NMX20b"]),
                ("time_period.end", "2020-10-07T20:28:00+00:00"),
                ("time_period.start", "2020-09-20T19:03:06+00:00"),
                ("transfer_function.coordinate_system", "geopgraphic"),
                ("transfer_function.id", "NMX20"),
                ("transfer_function.processed_date", None),
                ("transfer_function.processing_parameters", ["{type: None}"]),
                (
                    "transfer_function.remote_references",
                    [
                        "NMX20b",
                        "NMX20",
                        "NMW20",
                        "COR21",
                        "NMY21-NMX20b",
                        "NMX20",
                        "UTS18",
                    ],
                ),
                ("transfer_function.runs_processed", ["NMX20a", "NMX20b"]),
                ("transfer_function.sign_convention", "exp(+ i\\omega t)"),
                ("transfer_function.units", None),
            ]
        )

        self.assertDictEqual(
            meta_dict, self.tf_h5.station_metadata.to_dict(single=True)
        )

    def test_runs(self):

        for run1, run2 in zip(
            self.tf_h5.station_metadata.runs, self.tf_obj.station_metadata.runs
        ):
            with self.subTest(run1.id):
                run1.data_logger.firmware.author = None
                rd1 = run1.to_dict(single=True)
                rd1.pop("mth5_type")
                rd1.pop("hdf5_reference")

                rd2 = run2.to_dict(single=True)
                rd2.pop("mth5_type")
                rd2.pop("hdf5_reference")
                self.assertDictEqual(rd1, rd2)

    def test_channels(self):

        for run1, run2 in zip(
            self.tf_h5.station_metadata.runs, self.tf_obj.station_metadata.runs
        ):
            for ch1 in run1.channels:
                ch2 = run2.get_channel(ch1.component)
                with self.subTest(f"{run1.id}_{ch1.component}"):

                    chd1 = ch1.to_dict(single=True)
                    chd1.pop("mth5_type")
                    chd1.pop("hdf5_reference")

                    chd2 = ch2.to_dict(single=True)
                    chd2.pop("mth5_type")
                    chd2.pop("hdf5_reference")
                    self.assertDictEqual(chd1, chd2)

    def test_estimates(self):

        for estimate in [
            "transfer_function",
            "transfer_function_error",
            "inverse_signal_power",
            "residual_covariance",
        ]:

            with self.subTest(estimate):
                est1 = getattr(self.tf_obj, estimate)
                est2 = getattr(self.tf_h5, estimate)

                self.assertTrue((est1.to_numpy() == est2.to_numpy()).all())

    def test_period(self):
        self.assertTrue((self.tf_obj.period == self.tf_h5.period).all())

    def test_tf_summary(self):
        self.mth5_obj.tf_summary.clear_table()
        self.mth5_obj.tf_summary.summarize()

        with self.subTest("test shape"):
            self.assertEqual(self.mth5_obj.tf_summary.shape, (1,))
        true_dict = dict(
            [
                ("station", b"NMX20"),
                ("survey", b"CONUS_South"),
                ("latitude", 34.470528),
                ("longitude", -108.712288),
                ("elevation", 1940.05),
                ("tf_id", b"NMX20"),
                ("units", b"none"),
                ("has_impedance", True),
                ("has_tipper", True),
                ("has_covariance", True),
                ("period_min", 4.6545500000000004),
                ("period_max", 29127.110000000001),
            ]
        )
        for name in self.mth5_obj.tf_summary.dtype.names:
            if "reference" in name:
                continue
            with self.subTest(f"test {name}"):
                self.assertEqual(
                    self.mth5_obj.tf_summary.array[name][0], true_dict[name]
                )

    @classmethod
    def tearDownClass(self):
        self.mth5_obj.close_mth5()
        self.fn.unlink()
