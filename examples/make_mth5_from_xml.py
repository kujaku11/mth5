# -*- coding: utf-8 -*-
"""
Example script to make an MTH5 file from real data

Created on Mon Jun 22 12:20:59 2020

@author: jpeacock

This script interrogates a directory containing a collection of XML files and
reads them into an MTH5 object.
"""
# =============================================================================
# imports
# =============================================================================
import numpy as np
from xml.etree import cElementTree as et

from mth5 import mth5
from mth5.utils.pathing import ensure_is_path

# =============================================================================
# functions
# =============================================================================

# <XML HELPERS>
def is_a_station_xml(fn):
    return fn.count(".") == 1


def is_a_run_xml(fn):
    return fn.count(".") == 2


def is_a_channel_xml(fn):
    return fn.count(".") > 2


def read_xml(xml_file):
    """
    :param xml_fn: DESCRIPTION
    :type xml_fn: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    return et.parse(xml_file).getroot()


def collect_xml_fn(station, directory):
    """
    Get all the files associated with a station

    :param station: DESCRIPTION
    :type station: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    fn_list: the list of xml files

    """
    directory = ensure_is_path(directory)
    xml_file_list = [x.name for x in directory.glob("*.xml") if station in x.name]
    station_dict = {"station": None, "runs": {}}
    for xml_file in xml_file_list:
        if is_a_station_xml(xml_file):
            station_dict["station"] = xml_file
        elif is_a_run_xml(xml_file):
            run_letter = xml_file.split(".")[1]
            try:
                station_dict["runs"][f"{station}{run_letter}"]["fn"] = xml_file
            except KeyError:
                station_dict["runs"][f"{station}{run_letter}"] = {
                    "fn": xml_file,
                    "channels": {},
                }
        elif is_a_channel_xml(xml_file):
            name_list = xml_file.split(".")
            run_letter = name_list[1]
            component = name_list[-2]
            try:
                station_dict["runs"][f"{station}{run_letter}"]["channels"][
                    component
                ] = xml_file
            except KeyError:
                station_dict["runs"][f"{station}{run_letter}"] = {
                    "fn": None,
                    "channels": {component: xml_file},
                }
        else:
            continue

    return station_dict


# </XML HELPERS>


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

        # loop over channels
        for channel, channel_fn in run_dict["channels"].items():
            _, _, channel_type, component, _ = channel_fn.split(".")
            channel = run.add_channel(component, channel_type, np.random.rand(4096))
            channel.metadata.from_xml(read_xml(directory.joinpath(channel_fn)))
            channel.metadata.time_period.start = run.metadata.time_period.start
            channel.metadata.time_period.end = run.metadata.time_period.end
            channel.write_metadata()

    return new_station



xml_root = "path/to/metadata/files.xml"

mth5_filename = "from_xml.mth5"
if mth5_filename.exists():
    mth5_filename.unlink()
    print(f"--> Rmoved existing file {mth5_filename}")

# initialize mth5 object
mth5_obj = mth5.MTH5()
mth5_obj.open_mth5(mth5_filename, mode="a")

### add survey information
survey_element = read_xml(xml_root.joinpath("survey.xml"))

survey_obj = mth5_obj.survey_group
survey_obj.metadata.from_xml(survey_element)
survey_obj.write_metadata()

for station in ["FL001", "FL002"]:
    # add station
    new_station = add_station(station, xml_root, mth5_obj)
mth5_obj.close_mth5()

