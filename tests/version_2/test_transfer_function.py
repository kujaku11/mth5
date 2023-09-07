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
        self.tf_obj.read()

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
                    "citation_dataset.authors",
                    "Schultz, A., Pellerin, L., Bedrosian, P., Kelbert, A., Crosbie, J.",
                ),
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
                    "copyright.acknowledgement:The USMTArray-CONUS South campaign was carried out through a cooperative agreement between\nthe U.S. Geological Survey (USGS) and Oregon State University (OSU). A subset of 40 stations\nin the SW US were funded through NASA grant 80NSSC19K0232.\nLand permitting, data acquisition, quality control and field processing were\ncarried out by Green Geophysics with project management and instrument/engineering\nsupport from OSU and Chaytus Engineering, respectively.\nProgram oversight, definitive data processing and data archiving were provided\nby the USGS Geomagnetism Program and the Geology, Geophysics and Geochemistry Science Centers.\nWe thank the U.S. Forest Service, the Bureau of Land Management, the National Park Service,\nthe Department of Defense, numerous state land offices and the many private landowners\nwho permitted land access to acquire the USMTArray data.; copyright.conditions_of_use:All data and metadata for this survey are available free of charge and may be copied freely, duplicated and further distributed provided that this data set is cited as the reference, and that the author(s) contributions are acknowledged as detailed in the Acknowledgements. Any papers cited in this file are only for reference. There is no requirement to cite these papers when the data are used. Whenever possible, we ask that the author(s) are notified prior to any publication that makes use of these data.\n While the author(s) strive to provide data and metadata of best possible quality, neither the author(s) of this data set, nor IRIS make any claims, promises, or guarantees about the accuracy, completeness, or adequacy of this information, and expressly disclaim liability for errors and omissions in the contents of this file. Guidelines about the quality or limitations of the data and metadata, as obtained from the author(s), are included for informational purposes only.; copyright.release_status:Unrestricted Release",
                ),
                ("country", ["USA"]),
                ("datum", "WGS84"),
                ("geographic_name", "CONUS South"),
                ("id", "CONUS South"),
                ("name", None),
                ("northwest_corner.latitude", 0.0),
                ("northwest_corner.longitude", 0.0),
                ("project", "USMTArray"),
                ("project_lead.email", None),
                ("project_lead.organization", None),
                ("release_license", "CC0-1.0"),
                ("southeast_corner.latitude", 0.0),
                ("southeast_corner.longitude", 0.0),
                ("summary", "Magnetotelluric Transfer Functions"),
                ("time_period.end_date", "2020-10-07"),
                ("time_period.start_date", "2020-09-20"),
            ]
        )

        h5_meta_dict = self.tf_h5.survey_metadata.to_dict(single=True)
        try:
            h5_meta_dict.pop("mth5_type")
        except KeyError:
            pass
        try:
            h5_meta_dict.pop("hdf5_reference")
        except KeyError:
            pass

        self.assertDictEqual(meta_dict, h5_meta_dict)

    def test_station_metadta(self):

        meta_dict = OrderedDict(
            [
                ("acquired_by.author", "National Geoelectromagnetic Facility"),
                ("channels_recorded", ["ex", "ey", "hx", "hy", "hz"]),
                (
                    "comments",
                    "description:Magnetotelluric Transfer Functions; primary_data.filename:NMX20b_NMX20_NMW20_COR21_NMY21-NMX20b_NMX20_UTS18.png; attachment.description:The original used to produce the XML; attachment.filename:NMX20b_NMX20_NMW20_COR21_NMY21-NMX20b_NMX20_UTS18.zmm; site.data_quality_notes.comments.author:Jade Crosbie, Paul Bedrosian and Anna Kelbert; site.data_quality_notes.comments.value:great TF from 10 to 10000 secs (or longer)",
                ),
                ("data_type", "mt"),
                ("fdsn.id", "USMTArray.NMX20.2020"),
                ("geographic_name", "Nations Draw, NM, USA"),
                ("id", "NMX20"),
                ("location.datum", "WGS84"),
                ("location.declination.epoch", "2020.0"),
                ("location.declination.model", "WMM"),
                ("location.declination.value", 9.09),
                ("location.elevation", 1940.05),
                ("location.latitude", 34.470528),
                ("location.longitude", -108.712288),
                ("orientation.angle_to_geographic_north", 0.0),
                ("orientation.method", None),
                ("orientation.reference_frame", "geographic"),
                ("provenance.archive.name", None),
                ("provenance.creation_time", "2021-03-17T14:47:44+00:00"),
                ("provenance.creator.name", None),
                ("provenance.software.author", None),
                (
                    "provenance.software.name",
                    "EMTF File Conversion Utilities 4.0",
                ),
                ("provenance.software.version", None),
                ("provenance.submitter.author", "Anna Kelbert"),
                ("provenance.submitter.email", "akelbert@usgs.gov"),
                ("provenance.submitter.name", "Anna Kelbert"),
                (
                    "provenance.submitter.organization",
                    "U.S. Geological Survey, Geomagnetism Program",
                ),
                ("release_license", "CC0-1.0"),
                ("run_list", ["NMX20a", "NMX20b"]),
                ("time_period.end", "2020-10-07T20:28:00+00:00"),
                ("time_period.start", "2020-09-20T19:03:06+00:00"),
                ("transfer_function.coordinate_system", "geopgraphic"),
                ("transfer_function.data_quality.good_from_period", 5.0),
                ("transfer_function.data_quality.good_to_period", 29127.0),
                ("transfer_function.data_quality.rating.value", 5),
                ("transfer_function.id", "NMX20"),
                (
                    "transfer_function.processed_by.author",
                    "Jade Crosbie, Paul Bedrosian and Anna Kelbert",
                ),
                (
                    "transfer_function.processed_by.name",
                    "Jade Crosbie, Paul Bedrosian and Anna Kelbert",
                ),
                ("transfer_function.processed_date", "1980-01-01"),
                ("transfer_function.processing_parameters", []),
                (
                    "transfer_function.processing_type",
                    "Robust Multi-Station Reference",
                ),
                (
                    "transfer_function.remote_references",
                    ["NMW20", "COR21", "UTS18"],
                ),
                ("transfer_function.runs_processed", ["NMX20a", "NMX20b"]),
                ("transfer_function.sign_convention", "exp(+ i\\omega t)"),
                ("transfer_function.software.author", "Gary Egbert"),
                ("transfer_function.software.last_updated", "2015-08-26"),
                ("transfer_function.software.name", "EMTF"),
                ("transfer_function.software.version", None),
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
                try:
                    rd1.pop("mth5_type")
                except KeyError:
                    pass
                try:
                    rd1.pop("hdf5_reference")
                except KeyError:
                    pass

                rd2 = run2.to_dict(single=True)
                try:
                    rd2.pop("mth5_type")
                except KeyError:
                    pass
                try:
                    rd2.pop("hdf5_reference")
                except KeyError:
                    pass
                self.assertDictEqual(rd1, rd2)

    def test_channels(self):

        for run1, run2 in zip(
            self.tf_h5.station_metadata.runs, self.tf_obj.station_metadata.runs
        ):
            for ch1 in run1.channels:
                ch2 = run2.get_channel(ch1.component)
                with self.subTest(f"{run1.id}_{ch1.component}"):

                    chd1 = ch1.to_dict(single=True)
                    try:
                        chd1.pop("mth5_type")
                    except KeyError:
                        pass
                    try:
                        chd1.pop("hdf5_reference")
                    except KeyError:
                        pass

                    chd2 = ch2.to_dict(single=True)
                    try:
                        chd2.pop("mth5_type")
                    except KeyError:
                        pass
                    try:
                        chd2.pop("hdf5_reference")
                    except KeyError:
                        pass
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


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
