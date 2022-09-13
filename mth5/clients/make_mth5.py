# -*- coding: utf-8 -*-
"""
Updated on Wed Aug  25 19:57:00 2021

@author: jpeacock + tronan
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import pandas as pd

from obspy.clients import fdsn
from obspy import UTCDateTime
from obspy import read as obsread
from obspy.core.inventory import Inventory


from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment

from mth5.mth5 import MTH5
from mth5.timeseries import RunTS
from mth5.utils.mth5_logger import setup_logger

# =============================================================================


class MakeMTH5:
    """
    Make an MTH5 file from data archived at IRIS.  You will need to know the
    data you want before hand, and place that into a dataframe with columns

    - 'network'   --> FDSN Network code
    - 'station'   --> FDSN Station code
    - 'location'  --> FDSN Location code
    - 'channel'   --> FDSN Channel code
    - 'start'     --> Start time YYYY-MM-DDThh:mm:ss
    - 'end'       --> End time YYYY-MM-DDThh:mm:ss

    From this data frame data will be pulled from IRIS using Obspy tools.


    """

    def __init__(self, client="IRIS", mth5_version="0.2.0"):
        self.column_names = [
            "network",
            "station",
            "location",
            "channel",
            "start",
            "end",
        ]
        self.client = client
        self.mth5_version = mth5_version
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")

    def _validate_dataframe(self, df):
        """
        Validate data frame to conform to the desired format

        :param df: Input dataframe or file path
        :type df: :class:`pd.DataFrame`, str, Path
        :raises IOError: If input file path does not exist
        :raises ValueError: If input value is not a dataframe
        :return: validated dataframe
        :rtype: :class:`pd.DataFrame`

        """
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, (str, Path)):
                fn = Path(df)
                if not fn.exists():
                    raise IOError(f"File {fn} does not exist. Check path")
                df = pd.read_csv(fn)
                df = df.fillna("")
            else:
                raise ValueError(
                    f"Input must be a pandas.Dataframe not {type(df)}"
                )

        if df.columns.to_list() != self.column_names:
            raise ValueError(
                f"column names in file {df.columns} are not the expected {self.column_names}"
            )

        return df

    def _run_010(self, experiment, unique_list, streams, mth5_object):
        """
        Loop over stations for file version 0.1.0

        :param experiment: experiment metadata
        :type experiment: :class:`mt_metadata.timeseries.Experiment`
        :param unique_list: unique list of runs
        :type unique_list: dataframe
        :param streams: time series data
        :type streams: obspy.Stream
        :param mth5_object: mth5 object
        :type mth5_object: :class:`mth5.MTH5`

        """
        self._loop_stations(unique_list[0]["stations"], streams, mth5_object)

    def _run_020(self, experiment, unique_list, streams, mth5_object):
        survey_map = dict([(s.fdsn.network, s.id) for s in experiment.surveys])

        for survey_dict in unique_list:
            # get the mt survey id that maps to the fdsn network
            fdsn_network = survey_dict["network"]
            survey_id = survey_map[fdsn_network]
            survey_group = mth5_object.get_survey(survey_id)

            self._loop_stations(
                survey_dict["stations"],
                streams,
                mth5_object,
                survey_group,
                survey_id,
            )

    def _loop_stations(
        self, stations, streams, mth5_object, survey_group=None, survey_id=None
    ):

        for station_id in stations:
            msstreams = streams.select(station=station_id)
            trace_start_times = sorted(
                list(set([tr.stats.starttime.isoformat() for tr in msstreams]))
            )
            trace_end_times = sorted(
                list(set([tr.stats.endtime.isoformat() for tr in msstreams]))
            )
            if len(trace_start_times) != len(trace_end_times):
                raise ValueError(
                    f"Do not have the same number of start {len(trace_start_times)}"
                    f" and end times {len(trace_end_times)} from streams"
                )

            if self.mth5_version in ["0.1.0"]:
                mobj = mth5_object
                try:
                    run_list = mth5_object.get_station(station_id).groups_list
                except Exception:
                    self.logger.warning(
                        f"Unable to retrieve station {station_id} - if there is more than one survey requested, use version 0.2.0 instead of 0.1.0"
                    )
                    run_list = []

            elif self.mth5_version in ["0.2.0"]:
                mobj = survey_group
                run_list = mth5_object.get_station(
                    station_id, survey_id
                ).groups_list

            try:
                run_list.remove("Transfer_Functions")
            except:
                pass

            n_times = len(trace_start_times)

            # adding logic if there are already runs filled in
            if len(run_list) == n_times:
                for run_id, start, end in zip(
                    run_list, trace_start_times, trace_end_times
                ):
                    # add the group first this will get the already filled in
                    # metadata to update the run_ts_obj.
                    run_group = mobj.stations_group.get_station(
                        station_id
                    ).add_run(run_id)

                    # then get the streams an add existing metadata
                    run_stream = msstreams.slice(
                        UTCDateTime(start), UTCDateTime(end)
                    )
                    run_ts_obj = RunTS()
                    run_ts_obj.from_obspy_stream(
                        run_stream, run_group.metadata
                    )
                    run_group.from_runts(run_ts_obj)

            # if there is just one run
            elif len(run_list) == 1:
                if n_times > 1:
                    for run_id, times in enumerate(
                        zip(trace_start_times, trace_end_times), 1
                    ):
                        run_group = mobj.stations_group.get_station(
                            station_id
                        ).add_run(f"{run_id:03}")
                        run_stream = msstreams.slice(
                            UTCDateTime(times[0]), UTCDateTime(times[1])
                        )
                        run_ts_obj = RunTS()
                        run_ts_obj.from_obspy_stream(
                            run_stream, run_group.metadata
                        )
                        run_group.from_runts(run_ts_obj)

                elif n_times == 1:
                    run_group = mobj.stations_group.get_station(
                        station_id
                    ).add_run(run_list[0])
                    run_stream = msstreams.slice(
                        UTCDateTime(times[0]), UTCDateTime(times[1])
                    )
                    run_ts_obj = RunTS()
                    run_ts_obj.from_obspy_stream(
                        run_stream, run_group.metadata
                    )
                    run_group.from_runts(run_ts_obj)

            elif len(run_list) != n_times:
                # If there is more than one run and more or less than one trace per run (possibly)
                self.logger.warning(
                    "More or less runs have been requested by the user "
                    + "than are defined in the metadata. Runs will be "
                    + "defined but only the requested run extents contain "
                    + "time series data "
                    + "based on the users request."
                )
                for run_id, start, end in zip(
                    run_list, trace_start_times, trace_end_times
                ):
                    # add the group first this will get the already filled in
                    # metadata
                    for run in run_list:
                        run_group = mobj.stations_group.get_station(
                            station_id
                        ).get_run(run)

                        # Chekcs for start and end times of runs
                        run_start = run_group.metadata.time_period.start
                        run_end = run_group.metadata.time_period.end

                        # Create if statement that checks for start and end
                        # times in the run.
                        # Compares start and end times of runs
                        # to start and end times of traces. Packs runs based on
                        # time spans
                        if UTCDateTime(start) >= UTCDateTime(
                            run_start
                        ) and UTCDateTime(end) <= UTCDateTime(run_end):
                            run_stream = msstreams.slice(
                                UTCDateTime(start), UTCDateTime(end)
                            )
                            run_ts_obj = RunTS()
                            run_ts_obj.from_obspy_stream(
                                run_stream, run_group.metadata
                            )
                            run_group.from_runts(run_ts_obj)
                        else:
                            continue

            else:
                raise ValueError("Cannot add Run for some reason.")

    def _process_list(self, experiment, unique_list, streams, mth5_object):
        versionDict = {"0.1.0": self._run_010, "0.2.0": self._run_020}
        process_run = versionDict[self.mth5_version]
        process_run(experiment, unique_list, streams, mth5_object)

    def make_mth5_from_fdsnclient(
        self, df, path=None, client=None, interact=False
    ):
        """
        Make an MTH5 file from an FDSN data center

        :param df: DataFrame with columns

         - 'network'   --> FDSN Network code
         - 'station'   --> FDSN Station code
         - 'location'  --> FDSN Location code
         - 'channel'   --> FDSN Channel code
         - 'start'     --> Start time YYYY-MM-DDThh:mm:ss
         - 'end'       --> End time YYYY-MM-DDThh:mm:ss

        :type df: :class:`pandas.DataFrame`
        :param path: Path to save MTH5 file to, defaults to None
        :type path: string or :class:`pathlib.Path`, optional
        :param client: FDSN client name, defaults to "IRIS"
        :type client: string, optional
        :raises AttributeError: If the input DataFrame is not properly
         formatted an Attribute Error will be raised.
        :raises ValueError: If the values of the DataFrame are not correct a
         ValueError will be raised.
        :return: MTH5 file name
        :rtype: :class:`pathlib.Path`


        .. seealso:: https://docs.obspy.org/packages/obspy.clients.fdsn.html#id1

        .. note:: If any of the column values are blank, then any value will
         searched for.  For example if you leave 'station' blank, any station
         within the given start and end time will be returned.



        """

        if path is None:
            path = Path().cwd()
        else:
            path = Path(path)

        if client is not None:
            self.client = client

        df = self._validate_dataframe(df)

        unique_list = self.get_unique_networks_and_stations(df)
        if self.mth5_version in ["0.1.0"] and len(unique_list) != 1:
            raise AttributeError(
                "MTH5 supports one survey/network per container."
            )

        file_name = path.joinpath(self.make_filename(df))

        # initiate MTH5 file
        mth5_object = MTH5(file_version=self.mth5_version)
        mth5_object.open_mth5(file_name, "w")

        # read in inventory and streams
        inv, streams = self.get_inventory_from_df(df, self.client)

        # translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
        translator = XMLInventoryMTExperiment()
        experiment = translator.xml_to_mt(inv)

        # Updates experiment information based on time extent of streams
        # rather than time extent of inventory
        # experiment = translator.drop_runs(mth5_object, streams)

        mth5_object.from_experiment(experiment)

        self._process_list(experiment, unique_list, streams, mth5_object)

        if not interact:
            mth5_object.close_mth5()

            return file_name
        if interact:
            return mth5_object

    def get_inventory_from_df(self, df, client=None, data=True):
        """
        Get an :class:`obspy.Inventory` object from a
        :class:`pandas.DataFrame`

        :param df: DataFrame with columns

         - 'network'   FDSN Network code
         - 'station'   FDSN Station code
         - 'location'  FDSN Location code
         - 'channel'   FDSN Channel code
         - 'start'     Start time YYYY-MM-DDThh:mm:ss
         - 'end'       End time YYYY-MM-DDThh:mm:ss

        :type df: :class:`pandas.DataFrame`
        :param client: FDSN client
        :type client: string
        :param data: True if you want data False if you want just metadata,
         defaults to True
        :type data: boolean, optional
        :return: An inventory of metadata requested and data
        :rtype: :class:`obspy.Inventory` and :class:`obspy.Stream`

        .. seealso:: https://docs.obspy.org/packages/obspy.clients.fdsn.html#id1

        .. note:: If any of the column values are blank, then any value will
         searched for.  For example if you leave 'station' blank, any station
         within the given start and end time will be returned.

        """
        if client is not None:
            self.client = client

        df = self._validate_dataframe(df)

        # Use obspy to retrieve metadata
        client = fdsn.Client(self.client)

        # Create empty stream, inventory objects that will be populated below
        streams = obsread().clear()
        inv = Inventory(networks=[], source="MTH5")

        # Sort the values to be logically ordered
        df.sort_values(self.column_names[:-1])

        # Build an inventory by looping over the rows in the dataframe
        # Use .groupby to make looping simpler

        # To do: Currently loop over network, station, channel/location for
        # purpose of nesting the inventories within each other. Is there a
        # better way to do this without requiring 3 loops?

        # First, group the dataframe by network-epoch
        network_group = df.groupby(["network", "start", "end"])
        for net, net_DF in network_group:
            net_code = net[0]
            net_start = net[1]
            net_end = net[2]

            net_inv = client.get_stations(
                net_start, net_end, network=net_code, level="network"
            )
            returned_network = net_inv.networks[0]

            # For this network-epoch, group by network-station-start-end
            # This will group all loc.chans together for the station-epochs in this network-epoch
            station_group = net_DF.groupby(
                ["network", "station", "start", "end"]
            )
            for sta, sta_DF in station_group:
                sta_net = sta[0]
                sta_code = sta[1]
                sta_start = sta[2]
                sta_end = sta[3]

                sta_inv = client.get_stations(
                    sta_start,
                    sta_end,
                    network=sta_net,
                    station=sta_code,
                    level="station",
                )
                returned_sta = sta_inv.networks[0].stations[0]

                # No need to .groupby() for channel-level since we want to loop over each channel request for this
                # station-epoch
                for chan in sta_DF.itertuples():
                    cha_inv = client.get_stations(
                        chan.start,
                        chan.end,
                        network=chan.network,
                        station=chan.station,
                        loc=chan.location,
                        channel=chan.channel,
                        level="response",
                    )

                    # There may be multiple metadata epochs for a given channel-timespan requested
                    # and to keep the metadata for all epochs you need to add each one to the station object
                    for returned_chan in (
                        cha_inv.networks[0].stations[0].channels
                    ):
                        returned_sta.channels.append(returned_chan)

                    # Grab the data, if specified
                    if data:
                        streams = (
                            client.get_waveforms(
                                chan.network,
                                chan.station,
                                chan.location,
                                chan.channel,
                                UTCDateTime(chan.start),
                                UTCDateTime(chan.end),
                            )
                            + streams
                        )

                # Add the stations to the associated network
                returned_network.stations.append(returned_sta)

            # Add the network (with station and channel info) to the inventory
            inv.networks.append(returned_network)

        return inv, streams

    def get_df_from_inventory(self, inventory):
        """
        Create an data frame from an inventory object

        :param inventory: inventory object
        :type inventory: :class:`obspy.Inventory`
        :return: dataframe in proper format
        :rtype: :class:`pandas.DataFrame`

        """

        rows = []
        for network in inventory.networks:
            for station in network.stations:
                for channel in station.channels:
                    entry = (
                        network.code,
                        station.code,
                        channel.location_code,
                        channel.code,
                        channel.start_date,
                        channel.end_date,
                    )
                    rows.append(entry)

        return pd.DataFrame(rows, columns=self.column_names)

    def get_unique_networks_and_stations(self, df):
        """
        Get unique lists of networks, stations, locations, and channels from
        a given data frame.

        [{'network': FDSN code, "stations": [list of stations for network]}]

        :param df: request data frame
        :type df: :class:`pandas.DataFrame`
        :return: list of network dictionaries with
         [{'network': FDSN code, "stations": [list of stations for network]}]
        :rtype: list

        """
        unique_list = []
        net_list = df["network"].unique()
        for network in net_list:
            network_dict = {
                "network": network,
                "stations": df[df.network == network]
                .station.unique()
                .tolist(),
            }
            unique_list.append(network_dict)

        return unique_list

    def make_filename(self, df):
        """
        Make a filename from a data frame that is networks and stations

        :param df: request data frame
        :type df: :class:`pandas.DataFrame`
        :return: file name as network_01+stations_network_02+stations.h5
        :rtype: string

        """

        unique_list = self.get_unique_networks_and_stations(df)

        return (
            "_".join(
                [
                    f"{d['network']}_{'_'.join(d['stations'])}"
                    for d in unique_list
                ]
            )
            + ".h5"
        )

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
