# -*- coding: utf-8 -*-
"""
Tests for Metadata module

.. todo::
    * write tests for to/from_xml
    

Created on Tue Apr 28 18:08:40 2020

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

import unittest
import json
import pandas as pd
from collections import OrderedDict
from operator import itemgetter
from mth5 import metadata
from mth5.utils.exceptions import MTSchemaError

# =============================================================================
# Tests
# =============================================================================
class TestBase(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.base_object = metadata.Base()
        self.extra_name = 'ExtraAttribute'
        self.extra_v_dict = {'type': str,
                             'required': True,
                             'units': 'mv',
                             'style': 'controlled vocabulary',
                             'description':'test adding attribute',
                             'options': ['10', '12'],
                             'alias': ['other'],
                             'example': 'extra'}
        self.extra_value = 10
        
    def test_validate_name(self):
        self.assertEqual('name.test_case', 
                         self.base_object._validate_name('name/TestCase'))
        
        self.assertRaises(MTSchemaError,
                          self.base_object._validate_name,
                          '0Name/Test_case')
        
    def test_add_attribute(self):
        self.base_object.add_base_attribute(self.extra_name,
                                            self.extra_value,
                                            self.extra_v_dict)
        self.assertIsInstance(self.base_object.extra_attribute, 
                              self.extra_v_dict['type'])
        self.assertEqual(self.base_object.extra_attribute, '10')
        
        
    def test_validate_type(self):
        self.assertEqual(10.0, self.base_object._validate_type('10',
                                                               'float'))
        self.assertEqual(10, self.base_object._validate_type('10', int))
        self.assertEqual('10', self.base_object._validate_type(10, str))
        self.assertEqual(True, self.base_object._validate_type('true', bool))
        
        number_list = [10, '11', 12.6, '13.3']
        self.assertEqual([10, 11, 12, 13], 
                         self.base_object._validate_type(number_list, int))
        self.assertEqual([10., 11., 12.6, 13.3], 
                          self.base_object._validate_type(number_list, float))
        self.assertEqual(['10', '11', '12.6', '13.3'], 
                          self.base_object._validate_type(number_list, str))
        self.assertEqual([True, False], 
                         self.base_object._validate_type(['true', 'False'],
                                                         bool))
        
class TestLocation(unittest.TestCase):
    def setUp(self):
        self.lat_str = '40:20:10.15'
        self.lat_value = 40.336153
        self.lon_str = '-115:20:30.40'
        self.lon_value = -115.34178
        self.elev_str = '1234.5'
        self.elev_value = 1234.5
        self.location_object = metadata.Location()
        self.lat_fail_01 = '40:20.34556'
        self.lat_fail_02 = 96.78
        self.lon_fail_01 = '-112:23.3453'
        self.lon_fail_02 = 190.87

    def test_str_conversion(self):
        self.location_object.latitude = self.lat_str
        self.assertAlmostEqual(self.lat_value, 
                               self.location_object.latitude,
                               places=5)
        
        self.location_object.longitude = self.lon_str
        self.assertAlmostEqual(self.lon_value, 
                               self.location_object.longitude,
                               places=5)
        
        self.location_object.elevation = self.elev_str
        self.assertAlmostEqual(self.elev_value, 
                               self.location_object.elevation,
                               places=1)
        
    def test_fails(self):
        self.assertRaises(ValueError, 
                          self.location_object._assert_lat_value,
                          self.lat_fail_01)
        self.assertRaises(ValueError, 
                          self.location_object._assert_lat_value,
                          self.lat_fail_02)
        self.assertRaises(ValueError, 
                          self.location_object._assert_lon_value,
                          self.lon_fail_01)
        self.assertRaises(ValueError, 
                          self.location_object._assert_lon_value,
                          self.lon_fail_02)

class TestSurveyMetadata(unittest.TestCase):
    """
    test metadata in is metadata out
    """

    def setUp(self):
        self.maxDiff = None
        self.meta_dict = {'survey':
                          {'acquired_by.author': 'MT',
                           'acquired_by.comments': 'tired',
                           'archive_id': 'MT001',
                           'archive_network': 'EM',
                           'citation_dataset.doi': 'http://doi.####',
                           'citation_journal.doi': None,
                           'comments': None,
                           'country': None,
                           'datum': 'WGS84',
                           'geographic_name': 'earth',
                           'name': 'entire survey of the earth',
                           'northwest_corner.latitude': 80.0,
                           'northwest_corner.longitude': 179.9,
                           'project': 'EM-EARTH',
                           'project_lead.author': 'T. Lurric',
                           'project_lead.email': 'mt@mt.org',
                           'project_lead.organization': 'mt rules',
                           'release_license': 'CC-0',
                           'southeast_corner.latitude': -80.,
                           'southeast_corner.longitude': -179.9,
                           'summary': None,
                           'time_period.end_date': '1980-01-01',
                           'time_period.start_date': '2080-01-01'}}
                          
        self.meta_dict = {'survey': 
                          OrderedDict(sorted(self.meta_dict['survey'].items(),
                                             key=itemgetter(0)))}
    
        self.survey_object = metadata.Survey()
    
    def test_in_out_dict(self):
        self.survey_object.from_dict(self.meta_dict)
        self.assertDictEqual(self.meta_dict, self.survey_object.to_dict())
        
    def test_in_out_series(self):
        survey_series = pd.Series(self.meta_dict['survey'])
        self.survey_object.from_series(survey_series)
        self.assertDictEqual(self.meta_dict, self.survey_object.to_dict())
        
    def test_in_out_json(self):
        survey_json = json.dumps(self.meta_dict)
        self.survey_object.from_json((survey_json))
        self.assertDictEqual(self.meta_dict, self.survey_object.to_dict())
        
        survey_json = self.survey_object.to_json(nested=True)
        self.survey_object.from_json(survey_json)
        self.assertDictEqual(self.meta_dict, self.survey_object.to_dict())
        
    def test_start_date(self):
        self.survey_object.time_period.start_date = '2020/01/02'
        self.assertEqual(self.survey_object.time_period.start_date,
                         '2020-01-02')
        
        self.survey_object.start_date = '01-02-2020T12:20:30.450000+00:00'
        self.assertEqual(self.survey_object.time_period.start_date,
                         '2020-01-02')

    def test_end_date(self):
        self.survey_object.time_period.start_date = '2020/01/02'
        self.assertEqual(self.survey_object.time_period.start_date,
                         '2020-01-02')
        
        self.survey_object.start_date = '01-02-2020T12:20:30.45Z'
        self.assertEqual(self.survey_object.time_period.start_date,
                         '2020-01-02')
        
    def test_latitude(self):
        self.survey_object.southeast_corner.latitude = '40:10:05.123'
        self.assertAlmostEqual(self.survey_object.southeast_corner.latitude,
                               40.1680897,
                               places=5)
        
    def test_longitude(self):
        self.survey_object.southeast_corner.longitude = '-115:34:24.9786'
        self.assertAlmostEqual(self.survey_object.southeast_corner.longitude,
                               -115.57361, places=5)
        
class TestStationMetadata(unittest.TestCase):
    """
    test station metadata
    """
    
    def setUp(self):
        self.maxDiff = None
        self.station_object = metadata.Station()
        self.meta_dict = {'station':
                          {'acquired_by.author': 'mt',
                           'acquired_by.comments': None,
                           'archive_id': 'MT012',
                           'channel_layout': 'L',
                           'channels_recorded': ['Ex', 'Ey', 'Hx', 'Hy'],
                           'comments': None,
                           'data_type': 'MT',
                           'geographic_name': 'london',
                           'id': 'mt012',
                           'location.declination.comments': None,
                           'location.declination.model': 'WMM',
                           'location.declination.value': 12.3,
                           'location.elevation': 1234.,
                           'location.latitude': 10.0,
                           'location.longitude': -112.98,
                           'orientation.method': 'compass',
                           'orientation.reference_frame': 'geographic',
                           'provenance.comments': None,
                           'provenance.creation_time': '1980-01-01T00:00:00+00:00',
                           'provenance.log': None,
                           'provenance.software.author': 'test',
                           'provenance.software.name': 'name',
                           'provenance.software.version': '1.0a',
                           'provenance.submitter.author': 'name',
                           'provenance.submitter.email': 'test@here.org',
                           'provenance.submitter.organization': None,
                           'time_period.end': '1980-01-01T00:00:00+00:00',
                           'time_period.start': '1980-01-01T00:00:00+00:00'}}
              
        self.meta_dict = {'station': 
                          OrderedDict(sorted(self.meta_dict['station'].items(),
                                             key=itemgetter(0)))}
    def test_in_out_dict(self):
        self.station_object.from_dict(self.meta_dict)
        self.assertDictEqual(self.meta_dict, self.station_object.to_dict())
        
    def test_in_out_series(self):
        station_series = pd.Series(self.meta_dict['station'])
        self.station_object.from_series(station_series)
        self.assertDictEqual(self.meta_dict, self.station_object.to_dict())
        
    def test_in_out_json(self):
        survey_json = json.dumps(self.meta_dict)
        self.station_object.from_json((survey_json))
        self.assertDictEqual(self.meta_dict, self.station_object.to_dict())
        
        survey_json = self.station_object.to_json(nested=True)
        self.station_object.from_json(survey_json)
        self.assertDictEqual(self.meta_dict, self.station_object.to_dict())
        
    def test_start(self):
        self.station_object.time_period.start = '2020/01/02T12:20:40.4560Z'
        self.assertEqual(self.station_object.time_period.start, 
                         '2020-01-02T12:20:40.456000+00:00')
        
        self.station_object.time_period.start = '01/02/20T12:20:40.4560'
        self.assertEqual(self.station_object.time_period.start, 
                         '2020-01-02T12:20:40.456000+00:00')

    def test_end_date(self):
        self.station_object.time_period.end = '2020/01/02T12:20:40.4560Z'
        self.assertEqual(self.station_object.time_period.end, 
                         '2020-01-02T12:20:40.456000+00:00')
        
        self.station_object.time_period.end = '01/02/20T12:20:40.4560'
        self.assertEqual(self.station_object.time_period.end, 
                         '2020-01-02T12:20:40.456000+00:00')
        
    def test_latitude(self):
        self.station_object.location.latitude = '40:10:05.123'
        self.assertAlmostEqual(self.station_object.location.latitude,
                               40.1680897, places=5)
        
    def test_longitude(self):
        self.station_object.location.longitude = '-115:34:24.9786'
        self.assertAlmostEqual(self.station_object.location.longitude,
                               -115.57361, places=5)
        
    def test_declination(self):
        self.station_object.location.declination.value = '10.980'
        self.assertEqual(self.station_object.location.declination.value,
                         10.980)
        
class TestRun(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.meta_dict = {'run': 
                          {'acquired_by.author': 'MT guru',
                           'acquired_by.comments': 'lazy',
                           'channels_recorded_auxiliary': None,
                           'channels_recorded_electric': ['EX', 'EY'],
                           'channels_recorded_magnetic': ['HX', 'HY', 'HZ'],
                           'comments': 'Cloudy solar panels failed',
                           'data_logger.firmware.author': 'MT instruments',
                           'data_logger.firmware.name': 'FSGMT',
                           'data_logger.firmware.version': '12.120',
                           'data_logger.id': 'mt091',
                           'data_logger.manufacturer': 'T. Lurric',
                           'data_logger.model': 'Ichiban',
                           'data_logger.power_source.comments': 'rats',
                           'data_logger.power_source.id': '12',
                           'data_logger.power_source.type': 'pb acid',
                           'data_logger.power_source.voltage.end': 12.0,
                           'data_logger.power_source.voltage.start': 14.0,
                           'data_logger.timing_system.comments': 'solid',
                           'data_logger.timing_system.drift': .001,
                           'data_logger.timing_system.type': 'GPS',
                           'data_logger.timing_system.uncertainty': .000001,
                           'data_logger.type': 'broadband',
                           'data_type': 'mt',
                           'id': 'mt01a',
                           'provenance.comments': None,
                           'provenance.log': None,
                           'metadata_by.author': 'MT guru',
                           'metadata_by.comments': 'lazy',
                           'sample_rate': 256.0,
                           'time_period.end': '1980-01-01T00:00:00+00:00',
                           'time_period.start': '1980-01-01T00:00:00+00:00'}}
            
        self.meta_dict = {'run': 
                          OrderedDict(sorted(self.meta_dict['run'].items(),
                                             key=itemgetter(0)))}
        self.run_object = metadata.Run()
   
    def test_in_out_dict(self):
        self.run_object.from_dict(self.meta_dict)
        self.assertDictEqual(self.meta_dict, self.run_object.to_dict())
        
    def test_in_out_series(self):
        run_series = pd.Series(self.meta_dict['run'])
        self.run_object.from_series(run_series)
        self.assertDictEqual(self.meta_dict, self.run_object.to_dict())
        
    def test_in_out_json(self):
        survey_json = json.dumps(self.meta_dict)
        self.run_object.from_json((survey_json))
        self.assertDictEqual(self.meta_dict, self.run_object.to_dict())
        
        survey_json = self.run_object.to_json(nested=True)
        self.run_object.from_json(survey_json)
        self.assertDictEqual(self.meta_dict, self.run_object.to_dict())
        
    def test_start(self):
        self.run_object.time_period.start = '2020/01/02T12:20:40.4560Z'
        self.assertEqual(self.run_object.time_period.start, 
                         '2020-01-02T12:20:40.456000+00:00')
        
        self.run_object.time_period.start = '01/02/20T12:20:40.4560'
        self.assertEqual(self.run_object.time_period.start, 
                         '2020-01-02T12:20:40.456000+00:00')

    def test_end_date(self):
        self.run_object.time_period.end = '2020/01/02T12:20:40.4560Z'
        self.assertEqual(self.run_object.time_period.end, 
                         '2020-01-02T12:20:40.456000+00:00')
        
        self.run_object.time_period.end = '01/02/20T12:20:40.4560'
        self.assertEqual(self.run_object.time_period.end, 
                         '2020-01-02T12:20:40.456000+00:00')
        
    def test_n_channels(self):
        self.run_object.from_dict(self.meta_dict)
        self.assertEqual(self.run_object.n_channels, 5)
        
        self.run_object.channels_recorded_auxiliary = ['T', 'battery']
        self.assertEqual(self.run_object.n_channels, 7)

class TestChannel(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.channel_object = metadata.Channel()
        self.meta_dict = {'channel':
                          {'comments': 'great',
                           'component': 'Temperature',
                           'channel_number': 1,
                           'data_quality.rating.author': 'mt',
                           'data_quality.rating.method': 'ml',
                           'data_quality.rating.value': 4,
                           'data_quality.warnings': None,
                           'filter.applied': [True],
                           'filter.comments': 'test',
                           'filter.name': ['lowpass', 'counts2mv'],
                           'location.elevation': 1234.,
                           'location.latitude': 12.324,
                           'location.longitude': -112.03,
                           'measurement_azimuth': 0.0,
                           'measurement_tilt': 0.0,
                           'sample_rate': 256.0,
                           'sensor.id': '1244A',
                           'sensor.manufacturer': 'faraday',
                           'sensor.model': 'ichiban',
                           'sensor.type': 'diode',
                           'time_period.end': '1980-01-01T00:00:00+00:00',
                           'time_period.start': '1980-01-01T00:00:00+00:00',
                           'translated_azimuth': 0.0,
                           'translated_tilt': 0.0,
                           'type': 'auxiliary',
                           'units': 'celsius'}}
        
        self.meta_dict = {'channel': 
                          OrderedDict(sorted(self.meta_dict['channel'].items(),
                                             key=itemgetter(0)))}
        
    def test_in_out_dict(self):
        self.channel_object.from_dict(self.meta_dict)
        self.assertDictEqual(self.meta_dict, self.channel_object.to_dict())
        
    def test_in_out_series(self):
        channel_series = pd.Series(self.meta_dict['channel'])
        self.channel_object.from_series(channel_series)
        self.assertDictEqual(self.meta_dict, self.channel_object.to_dict())
        
    def test_in_out_json(self):
        survey_json = json.dumps(self.meta_dict)
        self.channel_object.from_json((survey_json))
        self.assertDictEqual(self.meta_dict, self.channel_object.to_dict())
        
        survey_json = self.channel_object.to_json(nested=True)
        self.channel_object.from_json(survey_json)
        self.assertDictEqual(self.meta_dict, self.channel_object.to_dict())
        
class TestElectric(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.electric_object = metadata.Electric()
        self.meta_dict = {'electric':
                          {'ac.end': 10.2,
                           'ac.start': 12.1,
                           'comments': None,
                           'component': 'EX',
                           'contact_resistance.end': 1.2,
                           'contact_resistance.start': 1.1,
                           'channel_number': 2,
                           'data_quality.rating.author': 'mt',
                           'data_quality.rating.method': 'ml',
                           'data_quality.rating.value': 4,
                           'data_quality.warnings': None,
                           'dc.end': 1.,
                           'dc.start': 2.,
                           'dipole_length': 100.0,
                           'filter.applied': [False],
                           'filter.comments': None,
                           'filter.name': ['counts2mv', 'lowpass'],
                           'measurement_azimuth': 90.0,
                           'measurement_tilt': 0.0,
                           'negative.elevation': 100.0,
                           'negative.id': 'a',
                           'negative.latitude': 12.12,
                           'negative.longitude': -111.12,
                           'negative.manufacturer': 'test',
                           'negative.model': 'fats',
                           'negative.type': 'pb-pbcl',
                           'positive.elevation': 101.,
                           'positive.id': 'b',
                           'positive.latitude': 12.123,
                           'positive.longitude': -111.14,
                           'positive.manufacturer': 'test',
                           'positive.model': 'fats',
                           'positive.type': 'ag-agcl',
                           'sample_rate': 256.0,
                           'time_period.end': '1980-01-01T00:00:00+00:00',
                           'time_period.start': '1980-01-01T00:00:00+00:00',
                           'translated_azimuth': 0.0,
                           'translated_tilt': 0.0,
                           'type': 'electric',
                           'units': 'counts'}}
        
        self.meta_dict = {'electric': 
                          OrderedDict(sorted(self.meta_dict['electric'].items(),
                                             key=itemgetter(0)))}
        
    def test_in_out_dict(self):
        self.electric_object.from_dict(self.meta_dict)
        self.assertDictEqual(self.meta_dict, self.electric_object.to_dict())
        
    def test_in_out_series(self):
        electric_series = pd.Series(self.meta_dict['electric'])
        self.electric_object.from_series(electric_series)
        self.assertDictEqual(self.meta_dict, self.electric_object.to_dict())
        
    def test_in_out_json(self):
        survey_json = json.dumps(self.meta_dict)
        self.electric_object.from_json((survey_json))
        self.assertDictEqual(self.meta_dict, self.electric_object.to_dict())
        
        survey_json = self.electric_object.to_json(nested=True)
        self.electric_object.from_json(survey_json)
        self.assertDictEqual(self.meta_dict, self.electric_object.to_dict())
        

class TestMagnetic(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.magnetic_object = metadata.Magnetic()
        self.meta_dict = {'magnetic':
                          {'measurement_azimuth': 0.0,
                           'measurement_tilt': 0.0,
                           'channel_number': 2,   
                           'component': 'hy', 
                           'data_quality.rating.author': 'mt',
                           'data_quality.rating.method': 'ml',
                           'data_quality.rating.value': 4,
                           'data_quality.warnings': None,
                           'location.elevation': 1230.9,
                           'filter.applied': [True],
                           'filter.name': ['counts2mv'],
                           'filter.comments': None,
                           'h_field_max.end': 12.3,
                           'h_field_max.start': 1200.1,
                           'h_field_min.end': 12.3,
                           'h_field_min.start': 1400.0,
                           'location.latitude': 40.234,
                           'location.longitude': -113.45,
                           'comments': None,
                           'sample_rate': 256.0,
                           'sensor.id': 'ant2284',
                           'sensor.manufacturer': None,
                           'sensor.type': 'induction coil',
                           'sensor.model': 'ant4',
                           'type': 'magnetic',
                           'units': 'mv',
                           'time_period.start': '1980-01-01T00:00:00+00:00',
                           'time_period.end': '1980-01-01T00:00:00+00:00',
                           'translated_azimuth': 0.0,
                           'translated_tilt': 0.0,}}
        
        self.meta_dict = {'magnetic': 
                          OrderedDict(sorted(self.meta_dict['magnetic'].items(),
                                             key=itemgetter(0)))}
        
    def test_in_out_dict(self):
        self.magnetic_object.from_dict(self.meta_dict)
        self.assertDictEqual(self.meta_dict, self.magnetic_object.to_dict())
        
    def test_in_out_series(self):
        magnetic_series = pd.Series(self.meta_dict['magnetic'])
        self.magnetic_object.from_series(magnetic_series)
        self.assertDictEqual(self.meta_dict, self.magnetic_object.to_dict())
        
    def test_in_out_json(self):
        survey_json = json.dumps(self.meta_dict)
        self.magnetic_object.from_json((survey_json))
        self.assertDictEqual(self.meta_dict, self.magnetic_object.to_dict())
        
        survey_json = self.magnetic_object.to_json(nested=True)
        self.magnetic_object.from_json(survey_json)
        self.assertDictEqual(self.meta_dict, self.magnetic_object.to_dict())
        
# =============================================================================
# run
# =============================================================================
if __name__ == '__main__':
    unittest.main()