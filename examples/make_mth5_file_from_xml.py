# -*- coding: utf-8 -*-
"""
Example script to make an MTH5 file from real data

Created on Mon Jun 22 12:20:59 2020

@author: jpeacock
"""
# =============================================================================
# imports
# =============================================================================
from pathlib import Path
from xml.etree import cElementTree as et
import numpy as np

from mth5 import mth5

# =============================================================================
# inputs
# =============================================================================
dir_path = Path(r"c:\Users\jpeacock\Documents\mt_format_examples\mth5")


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
    
    station_dict ={'station': None, 'runs': {}}
    for fn in fn_list:
        if fn.count('.') == 1:
            station_dict['station'] = fn
        elif fn.count('.') == 2:
            run_letter = fn.split('.')[1]
            try:
                station_dict['runs'][f'{station}{run_letter}']['fn'] = fn
            except KeyError:
                station_dict['runs'][f'{station}{run_letter}'] = {'fn':fn,
                                                             'channels':{}}
        elif fn.count('.') > 2:
            name_list = fn.split('.')
            run_letter = name_list[1]
            comp = name_list[-2]
            try:
                station_dict['runs'][f'{station}{run_letter}']['channels'][comp] = fn
            except KeyError:
                station_dict['runs'][f'{station}{run_letter}'] = {'fn': None, 
                                                         'channels':{comp: fn}}
        else:
            continue
        
    return station_dict

def add_station(station, directory):
    """
    
    :param station: DESCRIPTION
    :type station: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    
    station_dict = collect_xml_fn(station, directory)
    
    # add station
    new_station = mth5_obj.stations_group.add_station(station)
    new_station.metadata.from_xml(read_xml(directory.joinpath(
        station_dict['station'])))
    new_station.write_metadata()
    
    # loop over runs
    for run_key, run_dict in station_dict['runs'].items():
        run = new_station.add_run(f'{run_key}')
        run.metadata.from_xml(read_xml(directory.joinpath(run_dict['fn'])))
        run.write_metadata()
        
        # update table entry
        table_index = new_station.summary_table.locate('id', run_key)
        new_station.summary_table.add_row(run.table_entry, table_index)
        
        # loop over channels
        for channel, channel_fn in run_dict['channels'].items():
            _, _, c_type, comp, _ = channel_fn.split('.')
            channel = run.add_channel(comp, c_type, np.random.rand(4096))
            channel.metadata.from_xml(read_xml(directory.joinpath(channel_fn)))
            channel.metadata.time_period.start = run.metadata.time_period.start
            channel.metadata.time_period.end = run.metadata.time_period.end
            channel.write_metadata()
            
            # update table entry
            table_index = run.summary_table.locate('component', comp)
            run.summary_table.add_row(channel.table_entry, table_index)
            
    return new_station
    

# =============================================================================
# script 
# =============================================================================
# initialize mth5 object
mth5_obj = mth5.MTH5()
mth5_obj.open_mth5(dir_path.joinpath('example_02.mth5'), mode='a')

### add survey information
survey_element = read_xml(dir_path.joinpath('survey.xml'))
survey_obj = mth5_obj.survey_group
survey_obj.metadata.from_xml(survey_element)
survey_obj.write_metadata()

for station in ['FL001', 'FL002']:
    # add station
    new_station = add_station(station, dir_path)
    
    # add entry to summary table
    mth5_obj.stations_group.summary_table.add_row(new_station.table_entry)

mth5_obj.close_mth5()


