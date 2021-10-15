# -*- coding: utf-8 -*-
"""
Updated on Wed Aug  25 19:57:00 2021

@author: jpeacock + tronan
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from obspy.clients import fdsn
from obspy import UTCDateTime
from obspy import read as obsread
from obspy.core.inventory import Inventory

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment

from mth5.mth5 import MTH5
from mth5.timeseries import RunTS
# =============================================================================

class MakeMTH5:

    def make_mth5_from_fdsnclient(self, df, path=None, client="IRIS"):
        """
        networks, stations, channels, starttime, endtime,
        Create an MTH5 file by pulling data from IRIS.

        To do this we use Obspy to talk to the IRIS DMC through
        :class:`obspy.clients.fdsn.Client`

        :parameter list[str] of network codes: FDSN network code
        :parameter list[str] of station codes: FDSN station code
        :parameter dictionary of channel and location codes: FDSN channel code
        :parameter starttime: start date and time
        :type starttime: :class:`obspy.UTCDateTime`
        :parameter endtime: end date and time
        :type endtime: :class:`obspy.UTCDateTime`

        """
        if path is None:
            path = Path().cwd()
        else:
            path = Path(path)
        
        net_list, sta_list, loc_list, chan_list = self.unique_df_combo(df)
        if len(net_list) != 1:
            raise AttributeError('MTH5 supports one survey/network per container.')
        
        file_name = path.joinpath(f"{''.join(net_list)}_{'_'.join(sta_list)}.h5")

        # initiate MTH5 file
        m = MTH5()
        m.open_mth5(file_name, "w")
        
        # read in inventory and streams
        inv, streams = self.inv_from_df(df, client)
        print("Got Streams")
        # translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
        translator = XMLInventoryMTExperiment()
        experiment = translator.xml_to_mt(inv)
        print(experiment)
        m.from_experiment(experiment)
        
        # TODO: Add survey level when structure allows.
        for msta_id in sta_list:
            # get the streams for the given station
            msstreams = streams.select(station=msta_id)
            trace_start_times = sorted(list(set([tr.stats.starttime.isoformat() for tr in msstreams])))
            trace_end_times = sorted(list(set([tr.stats.endtime.isoformat() for tr in msstreams])))
            if len(trace_start_times) != len(trace_end_times):
                raise ValueError(
                    f"Do not have the same number of start {len(trace_start_times)}"
                    f" and end times {len(trace_end_times)} from streams")
            run_list = m.get_station(msta_id).groups_list
            print(f"runs: {run_list}")
            n_times = len(trace_start_times)
            print(n_times)
            
            if len(run_list) == n_times:
                for run_id, start, end in zip(run_list, trace_start_times, trace_end_times):
                    print("="*50)
                    print(run_id)
                    print("="*50)
                    run_stream = msstreams.slice(UTCDateTime(start), UTCDateTime(end))
                    run_ts_obj = RunTS()
                    run_ts_obj.from_obspy_stream(run_stream)
                    run_group = m.stations_group.get_station(msta_id).add_run(run_id)
                    run_group.from_runts(run_ts_obj)
            elif len(run_list) == 1:
                if n_times > 1:
                    for run_id, times in enumerate(zip(trace_start_times, trace_end_times), 1):
                        print("="*50)
                        print(run_id)
                        print("="*50)
                        run_stream = msstreams.slice(UTCDateTime(times[0]), 
                                                     UTCDateTime(times[1]))
                        run_ts_obj = RunTS()
                        run_ts_obj.from_obspy_stream(run_stream)
                        run_group = m.stations_group.get_station(msta_id).add_run(f"{run_id:03}")
                        run_group.from_runts(run_ts_obj)
                elif n_times == 1:
                    run_stream = msstreams.slice(UTCDateTime(times[0]), 
                                                 UTCDateTime(times[1]))
                    run_ts_obj = RunTS()
                    run_ts_obj.from_obspy_stream(run_stream)
                    run_group = m.stations_group.get_station(msta_id).add_run(run_list[0])
                    run_group.from_runts(run_ts_obj)
            else:
                print("failed")
                
        m.close_mth5()

        return experiment, file_name

    def inv_from_df(self, df, client):

        # get the metadata
        client = fdsn.Client(client)
        streams = obsread()
        streams.clear()
        inv = Inventory(networks=[], source="MTH5")
        net_list, sta_list, loc_list, chan_list = self.unique_df_combo(df)
        df.sort_values(['net', 'sta', 'loc', 'chan', 'startdate'])
        usednet = dict()
        usedsta = dict()
        for net_index, net_scle in df.iterrows():

            # First for loop buids out networks and stations
            network = net_scle['net']
            net_start = net_scle['startdate']
            net_end = net_scle['enddate']
            if network not in usednet:
                net_inv = client.get_stations(net_start, net_end, network=network,
                                              level="network")
                returned_network = net_inv.networks[0]
                usednet[network] = [net_start]
            elif usednet.get(network) is not None and net_start not in usednet.get(network):
                net_inv = client.get_stations(net_start, net_end, network=network,
                                              level="network")
                returned_network = net_inv.networks[0]
                usednet[network].append(net_start)
            else:
                continue
            for sta_index, n_sta_cle in df.iterrows():
                # Station Loop

                station = n_sta_cle['sta']
                sta_start = n_sta_cle['startdate']
                sta_end = n_sta_cle['enddate']
                if network != n_sta_cle['net']:
                    continue
                else:
                    if station not in usedsta:
                        sta_inv = client.get_stations(sta_start, sta_end, network=network, station=station, level="station")
                        returned_sta = sta_inv.networks[0].stations[0]
                        usedsta[station] = [sta_start]
                    elif usedsta.get(station) is not None and sta_start not in usedsta.get(station):
                        # Checks for epoch
                        sta_inv = client.get_stations(sta_start, sta_end, network=network, station=station, level="station")
                        returned_sta = sta_inv.networks[0].stations[0]
                        usedsta[station].append(sta_start)
                    else:
                        continue
                for chan_index, ns_cha_le in df.iterrows():
                    # Channel loop
                    net_channel = ns_cha_le['net']
                    sta_channel = ns_cha_le['sta']
                    location = ns_cha_le['loc']
                    channel = ns_cha_le['chan']
                    chan_start = ns_cha_le['startdate']
                    chan_end = ns_cha_le['enddate']
                    if net_channel == network and sta_channel == station and chan_start == sta_start:
                        cha_inv = client.get_stations(
                            chan_start, chan_end, network=network, station=station,
                            loc=location, channel=channel, level="response")
                        returned_chan = cha_inv.networks[0].stations[0].channels[0]
                        returned_sta.channels.append(returned_chan)
                        #  Builds streams and inventories at the same time.
                        if streams is None:
                            streams = client.get_waveforms(network, station, location,
                                                           channel,
                                                           UTCDateTime(chan_start),
                                                           UTCDateTime(chan_end))
                        else:
                            streams = client.get_waveforms(network, station,
                                                           location, channel,
                                                           UTCDateTime(chan_start),
                                                           UTCDateTime(chan_end)) + streams
                    else:
                        continue
                returned_network.stations.append(returned_sta)
            inv.networks.append(returned_network)
        return inv, streams

    def unique_df_combo(self, df):
        net_list = df['net'].unique()
        sta_list = df['sta'].unique()
        loc_list = df['loc'].unique()
        chan_list = df['chan'].unique()
        return net_list, sta_list, loc_list, chan_list

    def get_fdsn_channel_map(self):
        FDSN_CHANNEL_MAP = {}

        FDSN_CHANNEL_MAP["BQ2"] = "BQ1"
        FDSN_CHANNEL_MAP["BQ3"] = "BQ2"
        FDSN_CHANNEL_MAP["BQN"] = "BQ1"
        FDSN_CHANNEL_MAP["BQE"] = "BQ2"
        FDSN_CHANNEL_MAP["BQZ"] = "BQ3"
        FDSN_CHANNEL_MAP["BT1"] = "BF1"
        FDSN_CHANNEL_MAP["BT2"] = "BF2"
        FDSN_CHANNEL_MAP["BT3"] = "BF3"
        FDSN_CHANNEL_MAP["LQ2"] = "LQ1"
        FDSN_CHANNEL_MAP["LQ3"] = "LQ2"
        FDSN_CHANNEL_MAP["LT1"] = "LF1"
        FDSN_CHANNEL_MAP["LT2"] = "LF2"
        FDSN_CHANNEL_MAP["LT3"] = "LF3"
        FDSN_CHANNEL_MAP["LFE"] = "LF1"
        FDSN_CHANNEL_MAP["LFN"] = "LF2"
        FDSN_CHANNEL_MAP["LFZ"] = "LF3"
        FDSN_CHANNEL_MAP["LQE"] = "LQ1"
        FDSN_CHANNEL_MAP["LQN"] = "LQ2"
        return FDSN_CHANNEL_MAP
