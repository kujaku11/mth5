# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:44:44 2020

@author: jpeacock
"""

import unittest
from mth5.utils import translator
from mth5 import metadata

# =============================================================================
# 
# =============================================================================
class TestSurvey2Network(unittest.TestCase):
    def setUp(self):
        self.survey_obj = metadata.Survey()
        self.meta_dict = {'survey':
                          {'acquired_by.author': 'MT',
                           'acquired_by.comments': 'tired',
                           'archive_id': 'MT01',
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
                           'release_license': 'CC 0',
                           'southeast_corner.latitude': -80.,
                           'southeast_corner.longitude': -179.9,
                           'summary': None,
                           'time_period.end_date': '1980-01-01',
                           'time_period.start_date': '2080-01-01'}}
        self.survey_obj.from_dict(self.meta_dict)
        
    def test_survey_to_network(self):
        network_obj = translator.mt_survey_to_inventory_network(
            self.survey_obj)
        
        self.assertEqual(network_obj.code, self.survey_obj.archive_network)
        self.assertEqual(network_obj.comments, self.survey_obj.comments)
        self.assertEqual(network_obj.start_date, 
                         self.survey_obj.time_period.start_date)
        self.assertEqual(network_obj.end_date, 
                         self.survey_obj.time_period.end_date)
        self.assertEqual(network_obj.restricted_status, 
                         self.survey_obj.release_license)
        self.assertEqual(network_obj.operators[0].agency,
                         self.survey_obj.project_lead.organization)
        self.assertEqual(network_obj.operators[0].contacts[0],
                         self.survey_obj.project_lead.author)
        

class TestStationMetadata(unittest.TestCase):
    """
    test station metadata
    """
    
    def setUp(self):
        self.maxDiff = None
        self.station_obj = metadata.Station()
        self.meta_dict = {'station':
                          {'acquired_by.author': 'mt',
                           'acquired_by.comments': None,
                           'archive_id': 'MT012',
                           'channel_layout': 'L',
                           'channels_recorded': 'Ex, Ey, Hx, Hy',
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
                           'orientation.layout_rotation_angle': 0.0,
                           'orientation.method': 'compass',
                           'orientation.option': 'geographic orthogonal',
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
        self.station_obj.from_dict(self.meta_dict)
            
    def test_station_to_station(self):
        inv_station = translator.mt_station_to_inventory_station(
            self.station_obj)
        
        self.assertEqual(inv_station.latitude,
                         self.station_obj.location.latitude)
        self.assertEqual(inv_station.longitude,
                         self.station_obj.location.longitude)
        self.assertEqual(inv_station.elevation, 
                         self.station_obj.location.elevation)
        self.assertEqual(inv_station.start_date, 
                         self.station_obj.time_period.start)
        self.assertEqual(inv_station.end_date, 
                         self.station_obj.time_period.end)
        self.assertListEqual(inv_station.channels, 
                             self.station_obj.channels_recorded)
        self.assertEqual(inv_station.creation_date, 
                         self.station_obj.time_period.start)
        self.assertEqual(inv_station.termination_date, 
                         self.station_obj.time_period.end)
        self.assertEqual(inv_station.site.description, 
                         self.station_obj.geographic_name)
        
# =============================================================================
# 
# =============================================================================
class TestElectric2Inventory(unittest.TestCase):
    def setUp(self):
        self.electric_obj = metadata.Electric()
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
                           'data_quality.warning': None,
                           'dc.end': 1.,
                           'dc.start': 2.,
                           'dipole_length': 100.0,
                           'filter.applied': [False],
                           'filter.comments': None,
                           'filter.name': ['counts2mv', 'lowpass'],
                           'measurement_azimuth': 90.0,
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
                           'type': 'electric',
                           'units': 'counts'}}
        self.electric_obj.from_dict(self.meta_dict)
        
        self.run_dict = {'run': 
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
                           'sampling_rate': 256.0,
                           'time_period.end': '1980-01-01T00:00:00+00:00',
                           'time_period.start': '1980-01-01T00:00:00+00:00'}}

        self.run_obj = metadata.Run()
        self.run_obj.from_dict(self.run_dict)
        
    def test_to_inventory_channel(self):
        inv_channel = translator.mt_electric_to_inventory_channel(
            self.electric_obj, self.run_obj)
        
        self.assertEqual(inv_channel.latitude,
                         self.electric_obj.positive.latitude)
        self.assertEqual(inv_channel.longitude,
                         self.electric_obj.positive.longitude)
        self.assertEqual(inv_channel.elevation,
                         self.electric_obj.positive.elevation)
        self.assertEqual(inv_channel.depth,
                         self.electric_obj.positive.elevation)
        self.assertEqual(inv_channel.azimuth,
                         self.electric_obj.measurement_azimuth)
        self.assertEqual(inv_channel.dip,
                         self.electric_obj.measurement_tilt)
        self.assertEqual(inv_channel.calibration_units, 
                         self.electric_obj.units)
        
class TestToXML(unittest.TestCase):
    def setUp(self):
        self.to_stationxml = translator.MTToStationXML()
        
    def test_build(self):
        survey = metadata.Survey()
        survey.from_dict({'survey':
                          {'acquired_by.author': 'MT',
                           'acquired_by.comments': 'tired',
                           'archive_id': 'MT01',
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
                           'release_license': 'CC 0',
                           'southeast_corner.latitude': -80.,
                           'southeast_corner.longitude': -179.9,
                           'summary': None,
                           'time_period.end_date': '1980-01-01',
                           'time_period.start_date': '2080-01-01'}})
        
        self.to_stationxml.add_network(survey)
            
        station = metadata.Station()
        station.from_dict({'station':
                          {'acquired_by.author': 'mt',
                           'acquired_by.comments': None,
                           'archive_id': 'MT012',
                           'channel_layout': 'L',
                           'channels_recorded': 'Ex, Ey, Hx, Hy',
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
                           'orientation.layout_rotation_angle': 0.0,
                           'orientation.method': 'compass',
                           'orientation.option': 'geographic orthogonal',
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
                           'time_period.start': '1980-01-01T00:00:00+00:00'}})
        
        self.to_stationxml.add_station(station)
            
        channel = metadata.Electric()
        channel.from_dict({'electric':
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
                           'data_quality.warning': None,
                           'dc.end': 1.,
                           'dc.start': 2.,
                           'dipole_length': 100.0,
                           'filter.applied': [False],
                           'filter.comments': None,
                           'filter.name': ['counts2mv', 'lowpass'],
                           'measurement_azimuth': 90.0,
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
                           'type': 'electric',
                           'units': 'counts'}})
            
        run = metadata.Run()
        run.from_dict({'run': 
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
                       'sampling_rate': 256.0,
                       'time_period.end': '1980-01-01T00:00:00+00:00',
                       'time_period.start': '1980-01-01T00:00:00+00:00'}})
            
        self.to_stationxml.add_channel(channel, run, 'MT012')
        
        self.to_stationxml.to_stationxml('Test_station.xml')
            
        
        
        
# =============================================================================
# run        
# =============================================================================
if __name__ == '__main__':
    unittest.main()