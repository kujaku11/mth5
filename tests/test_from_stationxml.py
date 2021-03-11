# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:05:54 2021

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
import unittest
from pathlib import Path
from mth5 import mth5
from mt_metadata.timeseries import stationxml
from mt_metadata.utils import STATIONXML_01
fn_path = Path(__file__).parent


class TestFromStationXML01(unittest.TestCase):
    """
    test from a stationxml 
    """

    def setUp(self):
        self.translator = stationxml.XMLInventoryMTExperiment()
        self.experiment = self.translator.xml_to_mt(
            stationxml_fn=STATIONXML_01)

        self.fn = fn_path.joinpath("from_stationxml.h5")
        if self.fn.exists():
            self.fn.unlink()

        self.m = mth5.MTH5()
        self.m.open_mth5(self.fn)
        self.m.from_experiment(self.experiment, 0)

    def test_groups(self):
        self.assertEqual(self.m.has_group("Survey"), True)
        self.assertEqual(self.m.has_group("Survey/Stations"), True)
        self.assertEqual(self.m.has_group("Survey/Stations/CAS04"), True)
        self.assertEqual(self.m.has_group("Survey/Stations/CAS04/001"), True)
        self.assertEqual(self.m.has_group(
            "Survey/Stations/CAS04/001/ey"), True)
        self.assertEqual(self.m.has_group(
            "Survey/Stations/CAS04/001/hy"), True)

    def test_survey_metadata(self):
        self.assertEqual(self.m.survey_group.metadata.fdsn.network, "ZU")
        self.assertEqual(
            self.m.survey_group.metadata.time_period.start_date, "2020-01-01")
        self.assertEqual(
            self.m.survey_group.metadata.time_period.end_date, "2023-12-31")
        self.assertEqual(self.m.survey_group.metadata.summary,
                         "USMTArray South Magnetotelluric Time Series (USMTArray CONUS South-USGS)")
        self.assertEqual(self.m.survey_group.metadata.citation_dataset.doi,
                         "10.7914/SN/ZU_2020")

    def test_station_metadata(self):
        station_dict = {
                "acquired_by.author": None,
                "channels_recorded": [],
                "data_type": None,
                "fdsn.id": "CAS04",
                "geographic_name": "Corral Hollow, CA, USA",
                "hdf5_reference": "<HDF5 object reference>",
                "id": "CAS04",
                "location.declination.model": None,
                "location.declination.value": None,
                "location.elevation": 329.3875,
                "location.latitude": 37.633351,
                "location.longitude": -121.468382,
                "mth5_type": "Station",
                "orientation.method": None,
                "orientation.reference_frame": "geographic",
                "provenance.software.author": None,
                "provenance.software.name": None,
                "provenance.software.version": None,
                "provenance.submitter.author": None,
                "provenance.submitter.email": None,
                "provenance.submitter.organization": None,
                "run_list": [
                    "001"
                ],
                "time_period.end": "2020-07-13T21:46:12+00:00",
                "time_period.start": "2020-06-02T18:41:43+00:00"
        }
        
        m_station = self.m.get_station(station_dict["id"]).metadata
        for key, true_value in station_dict.items():
            self.assertEqual(true_value, m_station.get_attr_from_name(key))
            
    def test_run_metadata(self):
        run_dict = {
            "id": "001",
            "channels_recorded_electric": ["ey"],
            "channels_recorded_magnetic": ["hy"],
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00"}
        
        m_run = self.m.get_run("CAS04", run_dict["id"]).metadata
        for key, true_value in run_dict.items():
            self.assertEqual(true_value, m_run.get_attr_from_name(key))
        

    # def test_to_stationxml(self):
    #     ex = self.m.to_experiment()
    #     x = self.translator.mt_to_xml(tationxml_fn=r"c:\Users\jpeacock\stationxml_01.xml")

    def tearDown(self):
        self.m.close_mth5()
        self.fn.unlink()

# m.close_mth5()
