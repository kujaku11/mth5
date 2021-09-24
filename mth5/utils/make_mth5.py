# -*- coding: utf-8 -*-
"""
Updated on Wed Aug  25 19:57:00 2021

@author: jpeacock + tronan
"""

from obspy.clients import fdsn
from obspy import UTCDateTime
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.mth5 import MTH5
from mth5.timeseries import RunTS
import os


class MakeMTH5:

    def make_mth5_from_fdsnclient(
        networks, stations, starttime, endtime, path=None, client="IRIS"
    ):
        """
        Create an MTH5 file by pulling data from IRIS.

        To do this we use Obspy to talk to the IRIS DMC through
        :class:`obspy.clients.fdsn.Client`

        :parameter list[str] of network codes: FDSN network code
        :parameter list[str] of station codes: FDSN station code
        :parameter starttime: start date and time
        :type starttime: :class:`obspy.UTCDateTime`
        :parameter endtime: end date and time
        :type endtime: :class:`obspy.UTCDateTime`

        """
        if type(networks) != list:
            raise TypeError('Input network code(s) must be formatted as a list.')

        if type(stations) != list:
            raise TypeError('Input station code(s) must be formatted as a list.')

        if path is None:
            path = str(os.getcwd()) + "/"
        start = UTCDateTime(starttime)
        end = UTCDateTime(endtime)
        file_name = path + "".join(networks) + "_" + "".join(stations) + "_" + starttime + ".h5"
        # need to know network, station, start and end times before hand
        # initiate MTH5 file
        m = MTH5()
        m.open_mth5(r"" + file_name, "w")
        client = fdsn.Client(client)
        # get the metadata
        inv = Inventory(networks=[], source="MTH5")
        for network in networks:
            # Builds an obspy inventory from a fdsn client.
            net_inv = client.get_stations(start, end, network=network,
                                          level="network"
                                          )
            returned_network = net_inv.networks[0]
            for station in stations:
                sta_inv = client.get_stations(
                    start, end, network=network, station=station, level="response")
                returned_sta = sta_inv.networks[0].stations[0]
                returned_network.stations.append(returned_sta)
            inv.networks.append(returned_network)

        for i_station, station in enumerate(stations):
            # Builds a MTH5 from the inventory.

            # get the data
            streams = client.get_waveforms(network, station, None, None, start, end)

            # translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
            translator = XMLInventoryMTExperiment()
            experiment = translator.xml_to_mt(inv)
            # fill metadata
            m.from_experiment(experiment)
            station_group = m.get_station(station)

            # runs can be split into channels with similar start times and sample rates
            start_times = sorted(
                list(set([tr.stats.starttime.isoformat() for tr in streams]))
            )
            end_times = sorted(list(set([tr.stats.endtime.isoformat() for tr in streams])))
            run_metadata = experiment.surveys[0].stations[i_station].runs[0]
            for index, times in enumerate(zip(start_times, end_times), 1):
                run_stream = streams.slice(UTCDateTime(times[0]), UTCDateTime(times[1]))
                run_ts_obj = RunTS()
                run_ts_obj.from_obspy_stream(run_stream, run_metadata)
                run_group = station_group.add_run(f"{index:03}")
                run_group.from_runts(run_ts_obj)
        m.close_mth5()
        return file_name
