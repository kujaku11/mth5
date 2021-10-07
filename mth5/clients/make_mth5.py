# -*- coding: utf-8 -*-
"""
Updated on Wed Aug  25 19:57:00 2021

@author: jpeacock + tronan
"""

from obspy.clients import fdsn
from obspy import UTCDateTime
from obspy import read as obsread
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.mth5 import MTH5
from mth5.utils import fdsn_tools
from mth5.timeseries import RunTS
from mth5.timeseries import ChannelTS
import os


class MakeMTH5:

    def make_mth5_from_fdsnclient(
        self, df, path=None, client="IRIS"
    ):
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
        # if type(networks) != list:
        #    raise TypeError('Input network code(s) must be formatted as a list.')

        # if type(stations) != list:
        #    raise TypeError('Input station code(s) must be formatted as a list.')
        makemth5 = MakeMTH5()
        run_ts = RunTS()
        if path is None:
            path = str(os.getcwd()) + "/"
        net_list, sta_list, loc_list, chan_list = self.unique_df_combo(df)
        # start = UTCDateTime(starttime)
        # end = UTCDateTime(endtime)
        file_name = path + "".join(net_list) + "_" + "".join(sta_list) + ".h5"
        # need to know network, station, start and end times before hand
        # initiate MTH5 file
        m = MTH5()
        streams = obsread()
        streams.clear()
        m.open_mth5(r"" + file_name, "w")
        inv, streams = self.inv_from_df(df, client)
        # get the metadata
        # translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
        translator = XMLInventoryMTExperiment()
        experiment = translator.xml_to_mt(inv)

        # Loop through the expiemrent and not through the stationxml
        for msta_id in sta_list:
            m.from_experiment(experiment)
            run_list = m.get_station(msta_id).groups_list
            for mrun_id in run_list:
                run_ts = RunTS()
                runts_list = []
                chan_list = m.stations_group.get_station(msta_id).get_run(mrun_id).groups_list
                for mchan_id in chan_list:
                    mth5_chan = m.stations_group.get_station(msta_id)\
                                 .get_run(mrun_id).get_channel(mchan_id)
                    tmp_fdsn_chan = fdsn_tools.make_channel_code(mth5_chan.metadata)
                    # Map the channel but mth5 and ph5
                    fdsn_channel_map = self.get_fdsn_channel_map()
                    mstream_chan = fdsn_channel_map.get(tmp_fdsn_chan)
                    # Get FDSN Channe code from dictionary
                    channel_ts = ChannelTS()
                    try:
                        mtrace = streams.select(station=msta_id, channel=tmp_fdsn_chan)
                        for trace in mtrace:
                            channel_ts.from_obspy_trace(trace)
                    except IndexError:
                        mtrace = streams.select(station=msta_id, channel=mstream_chan)
                        for trace in mtrace:
                            channel_ts.from_obspy_trace(mtrace)
                    runts_list.append(channel_ts)
                run_ts.set_dataset(runts_list)

        return file_name

    def inv_from_df(self, df, client):
        # THe only input is a dataframe

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
                            streams = client.get_waveforms(network, station, location, channel, UTCDateTime(chan_start), UTCDateTime(chan_end))
                        else:
                            streams = client.get_waveforms(network, station, location, channel, UTCDateTime(chan_start), UTCDateTime(chan_end)) + streams
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
