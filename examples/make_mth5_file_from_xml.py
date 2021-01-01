# -*- coding: utf-8 -*-
"""
Example script to make an MTH5 file from real data

Created on Mon Jun 22 12:20:59 2020

@author: jpeacock

Context here is a multiple station MT survey in Florida, and we are going to add a station
to that survey.  The survey itself is being represented as an mth5 object 
(in this case a "SurveyGroup")
"""
# =============================================================================
# imports
# =============================================================================
import numpy as np
from pathlib import Path
from xml.etree import cElementTree as et

from mth5 import mth5
from mth5.utils.pathing import DATA_DIR

# =============================================================================
# functions
# =============================================================================
def read_xml(xml_fn):
    """
    
    :param xml_fn: DESCRIPTION
    :type xml_fn: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    return et.parse(xml_fn).getroot()


def collect_xml_fn(station, directory):
    """
    Get all the files associated with a station
    
    :param station: DESCRIPTION
    :type station: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    if not isinstance(directory, Path):
        directory = Path(directory)
    fn_list = [fn.name for fn in directory.glob("*.xml") if station in fn.name]

    station_dict = {"station": None, "runs": {}}
    for fn in fn_list:
        if fn.count(".") == 1:
            station_dict["station"] = fn
        elif fn.count(".") == 2:
            run_letter = fn.split(".")[1]
            try:
                station_dict["runs"][f"{station}{run_letter}"]["fn"] = fn
            except KeyError:
                station_dict["runs"][f"{station}{run_letter}"] = {
                    "fn": fn,
                    "channels": {},
                }
        elif fn.count(".") > 2:
            name_list = fn.split(".")
            run_letter = name_list[1]
            comp = name_list[-2]
            try:
                station_dict["runs"][f"{station}{run_letter}"]["channels"][comp] = fn
            except KeyError:
                station_dict["runs"][f"{station}{run_letter}"] = {
                    "fn": None,
                    "channels": {comp: fn},
                }
        else:
            continue

    return station_dict


def add_station(station, directory, h5_obj):
    """
    
    :param station: DESCRIPTION
    :type station: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    station_dict = collect_xml_fn(station, directory)

    # add station
    new_station = h5_obj.stations_group.add_station(station)
    new_station.metadata.from_xml(read_xml(directory.joinpath(station_dict["station"])))
    new_station.write_metadata()

    # loop over runs
    for run_key, run_dict in station_dict["runs"].items():
        run = new_station.add_run(f"{run_key}")
        run.metadata.from_xml(read_xml(directory.joinpath(run_dict["fn"])))
        run.write_metadata()

        # update table entry
        table_index = new_station.summary_table.locate("id", run_key)
        new_station.summary_table.add_row(run.table_entry, table_index)

        # loop over channels
        for channel, channel_fn in run_dict["channels"].items():
            _, _, channel_type, component, _ = channel_fn.split(".")
            channel = run.add_channel(component, channel_type, np.random.rand(4096))
            channel.metadata.from_xml(read_xml(directory.joinpath(channel_fn)))
            channel.metadata.time_period.start = run.metadata.time_period.start
            channel.metadata.time_period.end = run.metadata.time_period.end
            channel.write_metadata()

            # update table entry
            table_index = run.summary_table.locate("component", component)
            run.summary_table.add_row(channel.table_entry, table_index)
            h5_obj.stations_group.summary_table.locate

    return new_station


# =============================================================================
# script
# =============================================================================
# set xml directory
xml_root = DATA_DIR.joinpath("florida_xml_metadata_files")

mth5_filename = DATA_DIR.joinpath("from_xml.mth5")
if mth5_filename.exists():
    mth5_filename.unlink()
    print(f"--> Rmoved existing file {mth5_filename}")

# initialize mth5 object
mth5_obj = mth5.MTH5()
mth5_obj.open_mth5(mth5_filename, mode="a")

### add survey information
#standalone xml
survey_element = read_xml(xml_root.joinpath('survey.xml'))
survey_element = xml_to_dict(survey_element) #this obj is a dict "shaped the same as the attrs of h5"

#Adding the info from xml to our mth5 survey
#USer inputs info to the metadata class, the metadata class validates it!!!,
# and then the survey, or mth5 object updates based on the metadata validation
#in this sense, the metadata class is acting as a sort of gatekeeper for changing mth5
#info, such as survey info or etc.
survey_obj = mth5_obj.survey_group
#Probably want a watcher in the mth5 Group(), it watches for changes in metadata.
#then when (valid) changes in metadata are detected, the mth5 object updates.
#
#the metadata that is stored in the HDF5 file is stored in a dictionary of attributes
#


#The metadata (provided that it only updates by setattr and a few from_qqq() methods,
#then we could use decorators in mth5,
#e.g.
#survey_obj.metadata.from_xml(survey_element)
#-->
#survey_obj.metadata_from_xml(survey_element)
#def metadata_from_xml(survey_element)
#    self.metadata_from_xml(survey_element)
#    self.write_metadata()

#survey_obj.metadata_from_json(jsonstring)
#and that is an instance of a "metadata update function"
#that triggers write_metadata()

survey_obj.metadata.from_xml(survey_element)
survey_obj.write_metadata()

for station_id in ["FL001", "FL002"]:
    # add station
    new_station = add_station(station, xml_root, mth5_obj)

mth5_obj.close_mth5()
