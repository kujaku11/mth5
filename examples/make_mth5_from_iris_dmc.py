# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:47:27 2021

@author: jpeacock
"""

from obspy.clients import fdsn
from obspy import UTCDateTime

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.mth5 import MTH5
from mth5.timeseries import RunTS

network = "ZU"
station = "CAS04"
start = UTCDateTime("2020-06-02T18:41:43.000000Z")
end = UTCDateTime("2020-07-13T21:46:12.000000Z")

# need to know network, station, start and end times before hand
client = fdsn.Client("IRIS")

# get the data
streams = client.get_waveforms(network, station, None, None, start, end) 

# get the metadata
inventory = client.get_stations(start, end, network=network, station=station,
                                level="channel")
# translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
translator = XMLInventoryMTExperiment()
experiment = translator.xml_to_mt(inventory)

# initiate MTH5 file
m = MTH5()
m.open_mth5(r"from_iris_dmc.h5", "w")

# fill metadata
m.from_experiment(experiment)
station_group = m.get_station(station)

# runs can be split into channels with similar start times and sample rates
start_times = sorted(list(set([tr.stats.starttime.isoformat() for tr in streams])))
end_times = sorted(list(set([tr.stats.endtime.isoformat() for tr in streams])))

for index, times in enumerate(zip(start_times, end_times), 1):
    run_stream = streams.slice(UTCDateTime(times[0]), UTCDateTime(times[1]))
    run_ts_obj = RunTS()
    run_ts_obj.from_obspy_stream(run_stream)
    run_group = station_group.add_run(f"{index:03}")
    run_group.from_runts(run_ts_obj)
    
m.close_mth5()
    
    