# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:05:54 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
#  Imports
# =============================================================================
import unittest
from pathlib import Path
from mth5 import mth5
from mt_metadata.timeseries import stationxml
from mt_metadata import STATIONXML_01

fn_path = Path(__file__).parent
# =============================================================================


class TestFromStationXML01(unittest.TestCase):
    """
    test from a stationxml
    """

    @classmethod
    def setUpClass(self):
        self.translator = stationxml.XMLInventoryMTExperiment()
        self.experiment = self.translator.xml_to_mt(
            stationxml_fn=STATIONXML_01
        )

        self.fn = fn_path.joinpath("from_stationxml.h5")
        if self.fn.exists():
            self.fn.unlink()
        self.m = mth5.MTH5(file_version="0.1.0")
        self.m.open_mth5(self.fn)
        self.m.from_experiment(self.experiment, 0)

    def test_groups(self):

        with self.subTest("has Survey"):
            self.assertEqual(self.m.has_group("Survey"), True)
        with self.subTest("has SStations"):
            self.assertEqual(self.m.has_group("Survey/Stations"), True)
        with self.subTest("has CAS04"):
            self.assertEqual(self.m.has_group("Survey/Stations/CAS04"), True)
        with self.subTest("has run 001"):
            self.assertEqual(
                self.m.has_group("Survey/Stations/CAS04/001"), True
            )
        with self.subTest("has channel ey"):
            self.assertEqual(
                self.m.has_group("Survey/Stations/CAS04/001/ey"), True
            )
        with self.subTest("has channel hy"):
            self.assertEqual(
                self.m.has_group("Survey/Stations/CAS04/001/hy"), True
            )

    def test_survey_metadata(self):
        with self.subTest("has network ZU"):
            self.assertEqual(self.m.survey_group.metadata.fdsn.network, "ZU")
        with self.subTest("test start"):
            self.assertEqual(
                self.m.survey_group.metadata.time_period.start_date,
                "2020-01-01",
            )
        with self.subTest("test end"):
            self.assertEqual(
                self.m.survey_group.metadata.time_period.end_date, "2023-12-31"
            )
        with self.subTest("survey summary"):

            self.assertEqual(
                self.m.survey_group.metadata.summary,
                "USMTArray South Magnetotelluric Time Series (USMTArray CONUS South-USGS)",
            )
        with self.subTest("doi"):
            self.assertEqual(
                self.m.survey_group.metadata.citation_dataset.doi,
                "10.7914/SN/ZU_2020",
            )

    def test_station_metadata(self):
        station_dict = {
            "acquired_by.author": "none",
            "channels_recorded": [],
            "data_type": "BBMT",
            "fdsn.id": "CAS04",
            "geographic_name": "Corral Hollow, CA, USA",
            "hdf5_reference": "<HDF5 object reference>",
            "id": "CAS04",
            "location.declination.model": "WMM",
            "location.declination.value": 0.0,
            "location.elevation": 329.3875,
            "location.latitude": 37.633351,
            "location.longitude": -121.468382,
            "mth5_type": "Station",
            "orientation.method": None,
            "orientation.reference_frame": "geographic",
            "provenance.software.author": "none",
            "provenance.software.name": None,
            "provenance.software.version": None,
            "provenance.submitter.author": None,
            "provenance.submitter.email": None,
            "provenance.submitter.organization": None,
            "run_list": ["001"],
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

        m_station = self.m.get_station(station_dict["id"]).metadata
        for key, true_value in station_dict.items():
            with self.subTest(key):
                self.assertEqual(true_value, m_station.get_attr_from_name(key))

    def test_run_metadata(self):
        run_dict = {
            "id": "001",
            "channels_recorded_electric": ["ey"],
            "channels_recorded_magnetic": ["hy"],
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

        m_run = self.m.get_run("CAS04", run_dict["id"]).metadata
        for key, true_value in run_dict.items():
            with self.subTest(key):
                self.assertEqual(true_value, m_run.get_attr_from_name(key))

    def test_ey_metadata(self):
        ch_dict = {
            "component": "ey",
            "positive.id": "200402F",
            "positive.manufacturer": "Oregon State University",
            "positive.type": "electrode",
            "positive.model": "Pb-PbCl2 kaolin gel Petiau 2 chamber type",
            "negative.id": "2004020",
            "negative.manufacturer": "Oregon State University",
            "negative.type": "electrode",
            "negative.model": "Pb-PbCl2 kaolin gel Petiau 2 chamber type",
            "dipole_length": 92.0,
            "measurement_azimuth": 103.2,
            "type": "electric",
            "units": "digital counts",
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

        m_ch = self.m.get_channel("CAS04", "001", "ey").metadata
        for key, true_value in ch_dict.items():
            with self.subTest(key):
                self.assertEqual(true_value, m_ch.get_attr_from_name(key))

    def test_hy_metadata(self):
        ch_dict = {
            "component": "hy",
            "measurement_azimuth": 103.2,
            "type": "magnetic",
            "units": "digital counts",
            "sensor.manufacturer": "Barry Narod",
            "sensor.model": "fluxgate NIMS",
            "sensor.type": "Magnetometer",
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

        m_ch = self.m.get_channel("CAS04", "001", "hy").metadata
        for key, true_value in ch_dict.items():
            with self.subTest(msg=key):
                self.assertEqual(true_value, m_ch.get_attr_from_name(key))

    def test_filters(self):

        for f_name in self.experiment.surveys[0].filters.keys():
            with self.subTest(f_name):
                exp_filter = self.experiment.surveys[0].filters[f_name]
                h5_filter = self.m.survey_group.filters_group.to_filter_object(
                    f_name
                )

                self.assertTrue(exp_filter, h5_filter)

    @classmethod
    def tearDownClass(self):
        self.m.close_mth5()
        self.fn.unlink()
