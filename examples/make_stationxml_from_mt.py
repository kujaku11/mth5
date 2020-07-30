# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:10:48 2020

@author: jpeacock
"""
from mth5.utils import translator
from mth5 import metadata

to_stationxml = translator.MTToStationXML()
        
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
                   'time_period.end_date': '1980-01-01T00:00:00.000000Z',
                   'time_period.start_date': '2020-01-01T00:00:00.000000Z'}})

to_stationxml.add_network(survey)
    
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
                   'orientation.method': 'compass',
                   'orientation.reference_frame': 'geomagnetic',
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

to_stationxml.add_station(station)
    
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
    
to_stationxml.add_channel(channel, run, 'MT012')
    
channel = metadata.Channel()
channel.from_dict({'channel':
                   {'comments': 'great',
                    'component': 'Temperature',
                    'channel_number': 1,
                    'data_quality.rating.author': 'mt',
                    'data_quality.rating.method': 'ml',
                    'data_quality.rating.value': 4,
                    'data_quality.warning': None,
                    'filter.applied': [True],
                    'filter.comments': 'test',
                    'filter.name': ['lowpass', 'counts2mv'],
                    'location.elevation': 1234.,
                    'location.latitude': 12.324,
                    'location.longitude': -112.03,
                    'measurement_azimuth': 0,
                    'measurement_tilt': 0,
                    'sample_rate': 256.0,
                    'time_period.end': '1980-01-01T00:00:00+00:00',
                    'time_period.start': '1980-01-01T00:00:00+00:00',
                    'type': 'temperature',
                    'units': 'celsius'}})
    
to_stationxml.add_channel(channel, run, 'MT012')

to_stationxml.to_stationxml('Test_station.xml')
