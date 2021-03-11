# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:10:48 2020

@author: jpeacock
"""
from pathlib import Path
from mth5 import mth5
from mt_metadata import timeseries as metadata
from mt_metadata.timeseries import stationxml


def test_make_stationxml_from_mt():

    fn_path = Path(__file__).parent
    fn = fn_path.joinpath("example.h5")
    if fn.exists():
        fn.unlink()
        
    survey = metadata.Survey()
    survey.from_dict(
        {
            "survey": {
                "acquired_by.author": "MT",
                "acquired_by.comments": "tired",
                "archive_id": "MT01",
                "fdsn.network": "EM",
                "citation_dataset.doi": "http://doi.####",
                "citation_journal.doi": None,
                "comments": None,
                "country": None,
                "datum": "WGS84",
                "geographic_name": "earth",
                "name": "entire survey of the earth",
                "northwest_corner.latitude": 80.0,
                "northwest_corner.longitude": 179.9,
                "project": "EM-EARTH",
                "project_lead.author": "T. Lurric",
                "project_lead.email": "mt@mt.org",
                "project_lead.organization": "mt rules",
                "release_license": "CC-0",
                "southeast_corner.latitude": -80.0,
                "southeast_corner.longitude": -179.9,
                "summary": None,
                "time_period.end_date": "1980-01-01T00:00:00.000000Z",
                "time_period.start_date": "2020-01-01T00:00:00.000000Z",
            }
        }
    )

    station = metadata.Station()
    station.from_dict(
        {
            "station": {
                "acquired_by.author": "mt",
                "acquired_by.comments": None,
                "archive_id": "MT012",
                "channel_layout": "L",
                "comments": None,
                "data_type": "MT",
                "geographic_name": "london",
                "id": "mt012",
                "fdsn.id": "mt012",
                "location.declination.comments": None,
                "location.declination.model": "WMM",
                "location.declination.value": 12.3,
                "location.elevation": 1234.0,
                "location.latitude": 10.0,
                "location.longitude": -112.98,
                "orientation.method": "compass",
                "orientation.reference_frame": "geomagnetic",
                "provenance.comments": None,
                "provenance.creation_time": "1980-01-01T00:00:00+00:00",
                "provenance.log": None,
                "provenance.software.author": "test",
                "provenance.software.name": "name",
                "provenance.software.version": "1.0a",
                "provenance.submitter.author": "name",
                "provenance.submitter.email": "test@here.org",
                "provenance.submitter.organization": None,
                "time_period.end": "1980-01-01T00:00:00+00:00",
                "time_period.start": "1980-01-01T00:00:00+00:00",
            }
        }
    )

    run = metadata.Run()
    run.from_dict(
        {
            "run": {
                "acquired_by.author": "MT guru",
                "acquired_by.comments": "lazy",
                "channels_recorded_auxiliary": ["temperature"],
                "channels_recorded_electric": [],
                "channels_recorded_magnetic": [],
                "comments": "Cloudy solar panels failed",
                "data_logger.firmware.author": "MT instruments",
                "data_logger.firmware.name": "FSGMT",
                "data_logger.firmware.version": "12.120",
                "data_logger.id": "mt091",
                "data_logger.manufacturer": "T. Lurric",
                "data_logger.model": "Ichiban",
                "data_logger.power_source.comments": "rats",
                "data_logger.power_source.id": "12",
                "data_logger.power_source.type": "pb acid",
                "data_logger.power_source.voltage.end": 12.0,
                "data_logger.power_source.voltage.start": 14.0,
                "data_logger.timing_system.comments": "solid",
                "data_logger.timing_system.drift": 0.001,
                "data_logger.timing_system.type": "GPS",
                "data_logger.timing_system.uncertainty": 0.000001,
                "data_logger.type": "broadband",
                "data_type": "mt",
                "id": "mt01a",
                "provenance.comments": None,
                "provenance.log": None,
                "metadata_by.author": "MT guru",
                "metadata_by.comments": "lazy",
                "sampling_rate": 256.0,
                "time_period.end": "1980-01-01T00:00:00+00:00",
                "time_period.start": "1980-01-01T00:00:00+00:00",
            }
        }
    )

    channel = metadata.Auxiliary()
    channel.from_dict(
        {
            "auxiliary": {
                "comments": "great",
                "component": "temperature",
                "channel_number": 1,
                "data_quality.rating.author": "mt",
                "data_quality.rating.method": "ml",
                "data_quality.rating.value": 4,
                "data_quality.warning": None,
                "filter.applied": [True, False],
                "filter.comments": "test",
                "filter.name": ["lowpass", "counts2mv"],
                "location.elevation": 1234.0,
                "location.latitude": 12.324,
                "location.longitude": -112.03,
                "measurement_azimuth": 0,
                "measurement_tilt": 0,
                "sample_rate": 256.0,
                "time_period.end": "1980-01-01T00:00:00+00:00",
                "time_period.start": "1980-01-01T00:00:00+00:00",
                "type": "temperature",
                "units": "celsius",
            }
        }
    )


    m = mth5.MTH5()
    m.open_mth5(fn)
    survey_group = m.survey_group
    survey_group.metadata.from_dict(survey.to_dict())
    survey_group.write_metadata()
    station_group = m.add_station(station.id, station)
    run_group = station_group.add_run(run.id, run)
    run_group.add_channel(channel.component, "auxiliary",
                          None, channel_metadata=channel)

    translator = stationxml.XMLInventoryMTExperiment()
    xml_inventory = translator.mt_to_xml(m.to_experiment(),
                                         stationxml_fn=fn_path.joinpath("example_stationxml.xml"))


if __name__ == "__main__":
    test_make_stationxml_from_mt()
