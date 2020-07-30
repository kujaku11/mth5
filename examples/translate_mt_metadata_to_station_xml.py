# -*- coding: utf-8 -*-
"""
This is an example script of how to translate MT metadata into StationXML

The steps can be:
    1) Make xml, json, or csv files for each level of metadata for each 
       station (Survey, Station, Run, Channel)
    
    2) You can then load these into the corresponding metadata object and
       write the corresponding obspy.Inventory object:
           
           ============================ ==================================
           obspy.Inventory              MT.metadata
           ============================ ==================================
           Network                      Survey
           Station                      Station
           Channel                      Channel + Run
           ============================ ==================================
           
.. note:: All MT metadata that does not fit into the StationXML schema will
          be included as extra under the heading <ns#:key>value</ns#:key>. 
          We are still working on making sure this is preserved at the DMC
          and can be accessible when users need.
       
Created on Thu Jun 11 12:51:18 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)
    
:license: 
    MIT
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from xml.etree import ElementTree as et

from mth5 import metadata
from mth5.utils import stationxml_translator

# =============================================================================
# Inputs: xml file names for each level
# =============================================================================
# directory where xml files reside
xml_path = Path(r"c:\Users\jpeacock\Documents\mt_format_examples\mth5")

# filenames for each level
fn_mt_survey_xml = r"survey.xml"
fn_mt_station_xml = r"FL001.xml"
fn_mt_run_xml = r"FL001.a.xml"
fn_mt_electric_xml = r"FL001.a.electric.Ex.xml"
fn_mt_magnetic_xml = r"FL001.a.magnetic.Hx.xml"

# =============================================================================
# Translate from MT to StationXML
# =============================================================================
# Initialize the MT to StationXML object
mt2xml = stationxml_translator.MTToStationXML()

# add network
mt_survey = metadata.Survey()
mt_survey.from_xml(et.parse(xml_path.joinpath(fn_mt_survey_xml)).getroot()) 
mt2xml.add_network(mt_survey)

# add station
mt_station = metadata.Station()
mt_station.from_xml(et.parse(xml_path.joinpath(fn_mt_station_xml)).getroot())
mt2xml.add_station(mt_station)

# get run information
mt_run = metadata.Run()
mt_run.from_xml(et.parse(xml_path.joinpath(fn_mt_run_xml)).getroot())

# add electric channel
mt_electric = metadata.Electric()
mt_electric.from_xml(et.parse(xml_path.joinpath(fn_mt_electric_xml)).getroot())
mt2xml.add_channel(mt_electric, mt_run, mt_station.archive_id)

# add magnetic channel
mt_magnetic = metadata.Magnetic()
mt_magnetic.from_xml(et.parse(xml_path.joinpath(fn_mt_magnetic_xml)).getroot())
mt2xml.add_channel(mt_magnetic, mt_run, mt_station.archive_id)

# write StationXML file
mt2xml.to_stationxml(xml_path.joinpath('florida_test_stationXML_02.xml'))
