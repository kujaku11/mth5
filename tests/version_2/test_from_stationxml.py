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
from mt_metadata import STATIONXML_01

fn_path = Path(__file__).parent


class TestFromStationXML01(unittest.TestCase):
    """
    test from a stationxml
    """

    def setUp(self):
        self.translator = stationxml.XMLInventoryMTExperiment()
        self.experiment = self.translator.xml_to_mt(stationxml_fn=STATIONXML_01)
        self.experiment.surveys[0].id = "test"
        self.base_path = "Experiment/Surveys/test"
        self.fn = fn_path.joinpath("from_stationxml.h5")

        self.m = mth5.MTH5(file_version="0.2.0")
        self.m.open_mth5(self.fn)
        self.m.from_experiment(self.experiment)

    def test_groups(self):
        with self.subTest(name="survey"):
            self.assertEqual(self.m.has_group(self.base_path), True)
        with self.subTest(name="stations"):
            self.assertEqual(self.m.has_group(f"{self.base_path}/Stations"), True)
        with self.subTest(name="station CAS04"):
            self.assertEqual(self.m.has_group(f"{self.base_path}/Stations/CAS04"), True)
        with self.subTest(name="run 001"):
            self.assertEqual(
                self.m.has_group(f"{self.base_path}/Stations/CAS04/001"), True
            )
        with self.subTest(name="channel ey"):
            self.assertEqual(
                self.m.has_group(f"{self.base_path}/Stations/CAS04/001/ey"), True
            )
        with self.subTest(name="channel hy"):
            self.assertEqual(
                self.m.has_group(f"{self.base_path}/Stations/CAS04/001/hy"), True
            )

    def test_survey_metadata(self):
        sg = self.m.get_survey("test")
        with self.subTest(name="network"):
            self.assertEqual(sg.metadata.fdsn.network, "ZU")
        with self.subTest(name="start time"):
            self.assertEqual(sg.metadata.time_period.start_date, "2020-06-02")
        with self.subTest(name="end time"):
            self.assertEqual(sg.metadata.time_period.end_date, "2020-07-13")
        with self.subTest(name="summary"):
            self.assertEqual(
                sg.metadata.summary,
                "USMTArray South Magnetotelluric Time Series (USMTArray CONUS South-USGS)",
            )
        with self.subTest(name="doi"):
            self.assertEqual(sg.metadata.citation_dataset.doi, "10.7914/SN/ZU_2020")

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
            "run_list": ["001"],
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

        m_station = self.m.get_station(station_dict["id"], survey="test").metadata
        for key, true_value in station_dict.items():
            with self.subTest(name=key):
                self.assertEqual(true_value, m_station.get_attr_from_name(key))

    def test_run_metadata(self):
        run_dict = {
            "id": "001",
            "channels_recorded_electric": ["ey"],
            "channels_recorded_magnetic": ["hy"],
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

        m_run = self.m.get_run("CAS04", run_dict["id"], survey="test").metadata
        for key, true_value in run_dict.items():
            with self.subTest(name=key):
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
            "units": "millivolts per kilometer",
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

        m_ch = self.m.get_channel("CAS04", "001", "ey", survey="test").metadata
        for key, true_value in ch_dict.items():
            with self.subTest(name=key):
                self.assertEqual(true_value, m_ch.get_attr_from_name(key))

    def test_hy_metadata(self):
        ch_dict = {
            "component": "hy",
            "measurement_azimuth": 103.2,
            "type": "magnetic",
            "units": "nanotesla",
            "sensor.manufacturer": "Barry Narod",
            "sensor.model": "fluxgate NIMS",
            "sensor.type": "Magnetometer",
            "time_period.end": "2020-07-13T21:46:12+00:00",
            "time_period.start": "2020-06-02T18:41:43+00:00",
        }

        m_ch = self.m.get_channel("CAS04", "001", "hy", survey="test").metadata
        for key, true_value in ch_dict.items():
            with self.subTest(name=key):
                self.assertEqual(true_value, m_ch.get_attr_from_name(key))

    def tearDown(self):
        self.m.close_mth5()
        self.fn.unlink()
