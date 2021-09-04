# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:47:27 2021

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from obspy.clients import fdsn
from obspy import read, UTCDateTime
from obspy.core.inventory import read_inventory

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.mth5 import MTH5
from mth5.timeseries import RunTS

# =============================================================================
# Inputs
# =============================================================================
network = "ZU"
station = "CAS04"
start = UTCDateTime("2020-06-02T18:41:43.000000Z")
end = UTCDateTime("2020-07-13T21:46:12.000000Z")

# write a stationxml from the MTH5
to_stationxml = True

# keep the MTH5 file open to interact with from the console
interact = True

# if testing on local machine use local so you don't have to download each test
local = False
local_path = Path(r"c:\Users\peaco\Documents\test_data\miniseed_cas04")
save_local = True
h5_fn = local_path.joinpath("from_iris_dmc.h5")
if h5_fn.exists():
    h5_fn.unlink()

# =============================================================================
# get the station xml and data
if not local:
    # need to know network, station, start and end times before hand
    client = fdsn.Client("IRIS")

    # get the data
    streams = client.get_waveforms(network, station, None, None, start, end)

    # get the metadata, be sure the level is response to get all the filter
    # information.
    inventory = client.get_stations(
        start, end, network=network, station=station, level="response"
    )
    if save_local:
        streams.write(local_path.joinpath(f"{station}.mseed").as_posix(), "mseed")
        inventory.write(local_path.joinpath(f"{station}.xml").as_posix(), "stationxml")

# just to be explicit
if local:
    # get the data
    streams = read(local_path.joinpath(f"{station}.mseed").as_posix())

    # get the metadata
    inventory = read_inventory(local_path.joinpath(f"{station}.xml").as_posix())

# translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
translator = XMLInventoryMTExperiment()
experiment = translator.xml_to_mt(inventory)
run_metadata = experiment.surveys[0].stations[0].runs[0]

# initiate MTH5 file
m = MTH5()
m.open_mth5(h5_fn, "w")

# fill metadata
m.from_experiment(experiment)
station_group = m.get_station(station)

# runs can be split into channels with similar start times and sample rates
start_times = sorted(list(set([tr.stats.starttime.isoformat() for tr in streams])))
end_times = sorted(list(set([tr.stats.endtime.isoformat() for tr in streams])))

for index, times in enumerate(zip(start_times, end_times), 1):
    run_id = f"{index:03}"
    run_stream = streams.slice(UTCDateTime(times[0]), UTCDateTime(times[1]))
    run_ts_obj = RunTS()
    # need to add run metadata because in the stationxml the channel metadata
    # is only one entry for all similar channels regardless of their duration
    # so we need to make sure that propagates to the MTH5.
    run_ts_obj.from_obspy_stream(run_stream, run_metadata)
    run_ts_obj.run_metadata.id = run_id
    run_group = station_group.add_run(run_id)
    run_group.from_runts(run_ts_obj)

if to_stationxml:
    new_inv = translator.mt_to_xml(
        m.to_experiment(), stationxml_fn=local_path.joinpath(f"{station}_from_mth5.xml")
    )

if not interact:
    m.close_mth5()
