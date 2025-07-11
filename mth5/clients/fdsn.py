# -*- coding: utf-8 -*-
"""
FDSN
=========

Module for working with FDSN clients using Obspy

Created on Fri Feb  4 15:53:21 2022

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
import copy
import numpy as np
from obspy.clients.fdsn import Client as FDSNClient
import pandas as pd
import time

from gzip import BadGzipFile
from loguru import logger
import obspy

# from obspy.clients import fdsn
# from obspy import UTCDateTime
# from obspy import read as obsread
# from obspy import read_inventory
# from obspy.core.inventory import Inventory
from pathlib import Path

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from mth5.mth5 import MTH5
from mth5.timeseries import RunTS

# =============================================================================


class FDSN:
    def __init__(self, client="IRIS", mth5_version="0.2.0", **kwargs):
        self.logger = logger
        self.request_columns = [
            "network",
            "station",
            "location",
            "channel",
            "start",
            "end",
        ]
        self.client = client

        # parameters of hdf5 file
        self.h5_compression = "gzip"
        self.h5_compression_opts = 4
        self.h5_shuffle = True
        self.h5_fletcher32 = True
        self.h5_data_level = 1
        self.mth5_version = mth5_version

        for key, value in kwargs.items():
            setattr(self, key, value)

        # ivars
        self._streams = None

    @property
    def h5_kwargs(self):
        h5_params = dict(
            file_version=self.mth5_version,
            compression=self.h5_compression,
            compression_opts=self.h5_compression_opts,
            shuffle=self.h5_shuffle,
            fletcher32=self.h5_fletcher32,
            data_level=self.h5_data_level,
        )

        for key, value in self.__dict__.items():
            if key.startswith("h5"):
                h5_params[key[3:]] = value

        return h5_params

    def _validate_dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, (str, Path)):
                fn = Path(df)
                if not fn.exists():
                    raise IOError(f"File {fn} does not exist. Check path")
                df = pd.read_csv(fn)
                df = df.fillna("")
            else:
                raise ValueError(f"Input must be a pandas.Dataframe not {type(df)}")
        if df.columns.to_list() != self.request_columns:
            raise ValueError(
                f"column names in file {df.columns} are not the expected "
                f"{self.request_columns}"
            )
        return df

    @property
    def run_list_ne_stream_intervals_message(self):
        """note about not equal stream intervals"""
        return (
            "More or less runs have been requested by the user "
            "than are defined in the metadata. Runs will be "
            "defined but only the requested run extents contain "
            "time series data based on the users request."
        )

    def _loop_stations(self, stations, m, survey_group=None):
        """
        loop over stations
        """
        for station_id in stations:
            self.wrangle_runs_into_containers(m, station_id, survey_group=survey_group)

    def _run_010(self, unique_list, m, **kwargs):
        """
        kwargs are supported just to make this a general function that can be
        kept in a dict and used as in process_list

        Parameters
        ----------
        unique_list
        m
        kwargs

        Returns
        -------

        """
        station_list = unique_list[0]["stations"]
        self._loop_stations(station_list, m)

    def _run_020(self, unique_list, m, experiment=None):
        """
        mt_metadata translates mt survey id into survey id if it (which?) is
        provided which will be different from the fdsn network id, so we need
        to map the fdsn networks onto the survey id.

        Parameters
        ----------
        unique_list
        m
        experiment

        Returns
        -------

        """
        survey_map = dict([(s.fdsn.network, s.id) for s in experiment.surveys])

        for survey_dict in unique_list:
            # get the mt survey id that maps to the fdsn network
            fdsn_network = survey_dict["network"]
            survey_id = survey_map[fdsn_network]
            survey_group = m.get_survey(survey_id)
            stations_list = survey_dict["stations"]
            self._loop_stations(stations_list, m, survey_group=survey_group)

    def _process_list(self, experiment, unique_list, m):
        """
        Routes job to correct processing based on mth5_version
        Maintainable way to handle future file versions and send them to their
        own processing functions if needed

        Parameters
        ----------
        experiment
        unique_list
        m

        Returns
        -------

        """

        version_dict = {"0.1.0": self._run_010, "0.2.0": self._run_020}

        process_run = version_dict[self.mth5_version]
        process_run(unique_list, m, experiment=experiment)

    def get_run_list_from_station_id(self, m, station_id, survey_id=None):
        """
        ignored_groups created to address issue #153.  This might be better placed
        closer to the core of mth5.

        Parameters
        ----------
        m
        station_id

        Returns
        -------
        run_list: list of strings
        """
        ignored_groups = [
            "Fourier_Coefficients",
            "Transfer_Functions",
        ]
        run_list = m.get_station(station_id, survey_id).groups_list
        run_list = [x for x in run_list if x not in ignored_groups]
        return run_list

    def stream_boundaries(self, streams):
        """
        Identify start and end times of streams

        Parameters
        ----------
        streams: obspy.core.stream.Stream

        Returns
        -------

        """
        start_times = [tr.stats.starttime.isoformat() for tr in streams]
        start_times = sorted(list(set(start_times)))
        end_times = [tr.stats.endtime.isoformat() for tr in streams]
        end_times = sorted(list(set(end_times)))
        if len(start_times) != len(end_times):
            raise ValueError(
                f"Do not have the same number of start {len(start_times)}"
                f" and end times {len(end_times)} from streams"
            )
        start_times = [obspy.UTCDateTime(x) for x in start_times]
        end_times = [obspy.UTCDateTime(x) for x in end_times]
        return start_times, end_times

    def get_station_streams(self, station_id):
        """Get streams for a certain station"""
        return self._streams.select(station=station_id)

    def get_run_group(self, mth5_obj_or_survey, station_id, run_id):
        """
        This method is key to merging wrangle_runs_into_containers_v1 and
        wrangle_runs_into_containers_v2.
        Because a v1 mth5 object can get a survey group with the same method
        as can a v2 survey_group

        Thus we can replace
        run_group = m.stations_group.get_station(station_id).add_run(run_id)
        &
        run_group = survey_group.stations_group.get_station(station_id).add_run(run_id)
        with
        run_group = mth5_obj_or_survey.stations_group.get_station(station_id).add_run(run_id)
        Parameters
        ----------
        mth5_obj_or_survey: mth5.mth5.MTH5 or mth5.groups.survey.SurveyGroup

        Returns
        -------

        """
        run_group = mth5_obj_or_survey.stations_group.get_station(station_id).add_run(
            run_id
        )
        return run_group

    def pack_stream_into_run_group(self, run_group, run_stream):
        """"""
        run_ts_obj = RunTS()
        run_ts_obj.from_obspy_stream(run_stream, run_group.metadata)
        run_group.from_runts(run_ts_obj)

        return run_group

    def run_timings_match_stream_timing(self, run_group, stream_start, stream_end):
        """
        Checks start and end times in the run.
        Compares start and end times of runs to start and end times of traces.
        If True, will packs runs based on time spans.

        Parameters
        ----------
        run_group
        stream_start
        stream_end

        Returns
        -------

        """
        streams_and_run_timings_match = False
        run_start = run_group.metadata.time_period.start
        run_end = run_group.metadata.time_period.end
        cond1 = stream_start >= obspy.UTCDateTime(run_start)
        cond2 = stream_end <= obspy.UTCDateTime(run_end)
        if cond1 and cond2:  # paired up
            streams_and_run_timings_match = True
        return streams_and_run_timings_match

    def wrangle_runs_into_containers(self, m, station_id, survey_group=None):
        """
        Note 1: There used to be two separate functions for this, but now there
        is one run_group_source is defined as either m or survey_group depending
        on v0.1.0 or 0.2.0

        Note 2: If/elif/elif/else Logic:
        The strategy is to add the group first. This will get the already filled
        in metadata to update the run_ts_obj. Then get streams an add existing
        metadata.


        Parameters
        ----------
        m
        streams
        station_id
        survey_group

        Returns
        -------

        """
        if survey_group is not None:
            survey_id = survey_group.metadata.id
            run_group_source = survey_group
        else:
            survey_id = None
            run_group_source = m
        # get the streams for the given station
        msstreams = self.get_station_streams(station_id)
        trace_start_times, trace_end_times = self.stream_boundaries(msstreams)
        run_list = self.get_run_list_from_station_id(m, station_id, survey_id=survey_id)
        num_streams = len(trace_start_times)

        # See Note 2
        # If number of runs and number of streams are the same, then metadata
        # matches the data and an easy pack.
        if len(run_list) == num_streams:
            for run_id, start, end in zip(run_list, trace_start_times, trace_end_times):
                run_group = self.get_run_group(run_group_source, station_id, run_id)
                run_stream = msstreams.slice(start, end)
                self.pack_stream_into_run_group(run_group, run_stream)

        # if the metadata contains only one run but there are multiple streams
        # then there is missing metadata that we need to add logically.  Add
        # runs sequentially and use metadata from the first run.
        elif len(run_list) == 1:
            self.logger.warning(
                "Only one run in the StationXML, but multiple runs identified "
                "from the data. Using first run metadata and channel metadata "
                "for the other channels and runs except time periods."
            )
            og_run_group = self.get_run_group(run_group_source, station_id, run_list[0])
            for run_num, times in enumerate(zip(trace_start_times, trace_end_times), 1):
                start = times[0]
                end = times[1]
                run_id = f"{run_num:03}"
                run_group = self.get_run_group(run_group_source, station_id, run_id)
                if run_num > 1:
                    # cleaner, but not working
                    og_run_group_metadata_dict = og_run_group.metadata.to_dict()
                    for key in ["id", "time_period.start", "time_period.end"]:
                        og_run_group_metadata_dict["run"].pop(key)
                    run_group.metadata.from_dict(og_run_group_metadata_dict)
                    run_group.write_metadata()

                run_stream = msstreams.slice(start, end)
                run_group = self.pack_stream_into_run_group(run_group, run_stream)

                # update channels from run 1 metadata
                if run_num > 1:
                    for ch in run_group.groups_list:
                        og_ch = og_run_group.get_channel(ch)
                        og_ch_metadata_dict = og_ch.metadata.to_dict(single=True)
                        # skip the start and end times
                        for key in ["time_period.start", "time_period.end"]:
                            og_ch_metadata_dict.pop(key)

                        new_ch = run_group.get_channel(ch)
                        new_ch.metadata.from_dict(og_ch_metadata_dict)
                        new_ch.write_metadata()

                    run_group.update_metadata()

        # If the number of runs does not equal the number of streams then
        # there is missing data or metadata.
        elif len(run_list) != num_streams:
            self.logger.warning(self.run_list_ne_stream_intervals_message)
            for start, end in zip(trace_start_times, trace_end_times):
                for run in run_list:
                    run_group = self.get_run_group(run_group_source, station_id, run)
                    if self.run_timings_match_stream_timing(run_group, start, end):
                        run_stream = msstreams.slice(start, end)
                        self.pack_stream_into_run_group(run_group, run_stream)
                        break
                    else:
                        continue
        else:
            raise ValueError("Cannot add Run for some reason.")
        return

    def make_mth5_from_fdsn_client(self, df, path=None, client=None, interact=False):
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

        if client is not None:
            self.client = client
        df = self._validate_dataframe(df)

        unique_list = self.get_unique_networks_and_stations(df)
        if self.mth5_version in ["0.1.0"]:
            if len(unique_list) != 1:
                raise AttributeError("MTH5 supports one survey/network per container.")

        # read in inventory and streams
        inv, streams = self.get_inventory_from_df(df, self.client)
        if interact:
            self.logger.warning(
                "Interact is deprecated.  Open the returned file path. \n\t"
                "> with MTH5() as m:\n\t\tm.open_mth5(filepath)\n\t\tdo something."
            )
        return self.make_mth5_from_inventory_and_streams(inv, streams, save_path=path)

    @property
    def streams(self):
        """obspy.Stream object"""
        return self._streams

    @streams.setter
    def streams(self, streams):
        """set streams can be a list of filenames"""

        if not isinstance(streams, obspy.Stream):
            if isinstance(streams, (list, tuple)):
                if not isinstance(streams[0], obspy.Stream):
                    if isinstance(streams[0], (str, Path)):
                        stream_list = obspy.read()
                        for fn in streams:
                            stream_list += obspy.read(fn)
                        self._streams = stream_list
                    else:
                        raise TypeError("Cannot understand streams input.")
        else:
            self._streams = streams

    def make_mth5_from_inventory_and_streams(self, inventory, streams, save_path=None):
        """
        Create an MTH5 from existing inventory and a stationXML

        inventory can be an obspy.Inventory object or a string or path to a StationXML

        streams must be a Stream object or a list of paths

        """

        if not isinstance(inventory, obspy.Inventory):
            if isinstance(inventory, (str, Path)):
                inventory = obspy.read_inventory(inventory)
            else:
                raise TypeError(f"Cannot understand inventory type {type(inventory)}")

        if save_path is None:
            save_path = Path().cwd()
        else:
            save_path = Path(save_path)

        self.streams = streams
        # translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
        translator = XMLInventoryMTExperiment()
        experiment = translator.xml_to_mt(inventory)

        retrieved_df = self.get_df_from_inventory(inventory)
        retrieved_unique_list = self.get_unique_networks_and_stations(retrieved_df)
        file_name = save_path.joinpath(self.make_filename(retrieved_df))

        # initiate MTH5 file
        with MTH5(**self.h5_kwargs) as m:
            m.open_mth5(file_name, "w")

            m.from_experiment(experiment)
            self._process_list(experiment, retrieved_unique_list, m)

            return m.filename

    def build_network_dict(self, df, client):
        """
        Build out a dictionary of networks, keyed by network_id, start_time.
        We could return this dict and use it as an auxilliary variable, but it seems easier to just add a column to
        the df.

        Parameters
        ----------
        df: pd.DataFrame
            This is a "request_df"

        Returns
        -------

        """
        # Build the dictionary
        networks = {}
        for row in df.itertuples():
            # First for loop builds out networks and stations
            if row.network not in networks.keys():
                networks[row.network] = {}
                net_inv = _fdsn_client_get_inventory(
                    client, row, response_level="network"
                )
                networks[row.network][row.start] = net_inv.networks[0]
            elif networks.get(row.network) is not None:
                if row.start not in networks[row.network].keys():
                    net_inv = _fdsn_client_get_inventory(
                        client, row, response_level="network"
                    )
                    networks[row.network][row.start] = net_inv.networks[0]
            else:
                continue
            if len(net_inv.networks) != 1:
                msg = (
                    f"Expected a unique network associated with {row.start}--{row.end}"
                )
                msg += f"Instead found {len(net_inv.networks)} networks"
                raise NotImplementedError(msg)
        return networks

    # def add_network_objects_to_request_df(self, df):
    #     networks_dict = self.build_network_dict(df, client)
    #     network_column = [networks[x.netork][x.start] for x in df.itertuples()]
    #     df["network_object"] = network_column
    #     return df

    def build_station_dict(self, df, client, networks_dict):
        """
        Given the {network-id, starttime}-keyed dict of networks, we build a station layer below this

        Parameters
        ----------
        df
        networks_dict

        Returns
        -------

        """
        stations_dict = copy.deepcopy(networks_dict)
        for network_id in networks_dict.keys():
            for start_time in networks_dict[network_id].keys():
                stations_dict[network_id][start_time] = {}
                cond1 = df.network == network_id
                cond2 = df.start == start_time
                sub_df = df[cond1 & cond2]
                sub_df.drop_duplicates("station", inplace=True)
                sub_df.reset_index(inplace=True, drop=True)

                for station_row in sub_df.itertuples():
                    sta_inv = _fdsn_client_get_inventory(
                        client,
                        station_row,
                        response_level="station",
                        max_tries=10,
                    )

                    stations_dict[network_id][start_time][station_row.station] = (
                        sta_inv.networks[0].stations[0]
                    )
        return stations_dict

    def get_waveforms_from_request_row(self, client, row):
        """

        Parameters
        ----------
        client
        row

        Returns
        -------

        """
        start = obspy.UTCDateTime(row.start)
        end = obspy.UTCDateTime(row.end)
        streams = client.get_waveforms(
            row.network, row.station, row.location, row.channel, start, end
        )
        return streams

    def get_inventory_from_df(self, df, client=None, data=True, max_tries=10):
        """
        20230806: The nested for looping here can make debugging complex, as well as lead to a lot of redundancies.
        I propose that we build out a dictionary of networks, keyed by network_id, start_time.
        It may actually be simpler to just add a column to the request_df that has the network_obj

        networks = {}
        networks[network_id] = {}
        networks[network_id][start_time_1] = obspy_network_obj
        networks[network_id][start_time_2] = obspy_network_obj
        ...

        Then the role of "returned_network" can be replaced by accessing the appropriate element and the second for-loop
        can move up by a layer of indentation.


        Will try to factor i
        Get an :class:`obspy.Inventory` object from a
        :class:`pandas.DataFrame`

        :param df: DataFrame with columns

            - 'network'   --> FDSN Network code
            - 'station'   --> FDSN Station code
            - 'location'  --> FDSN Location code
            - 'channel'   --> FDSN Channel code
            - 'start'     --> Start time YYYY-MM-DDThh:mm:ss
            - 'end'       --> End time YYYY-MM-DDThh:mm:ss

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

        # get the metadata from an obspy client
        client = FDSNClient(self.client)

        # creat an empty stream to add to
        streams = obspy.read()
        streams.clear()

        inv = obspy.Inventory(networks=[], source="MTH5")

        # sort the values to be logically ordered
        df.sort_values(self.request_columns[:-1])

        # Build helper dictionares of networks and stations
        networks_dict = self.build_network_dict(df, client)
        stations_dict = self.build_station_dict(df, client, networks_dict)

        # Pack channels into stations
        for ch_row in df.itertuples():
            station_obj = stations_dict[ch_row.network][ch_row.start][ch_row.station]
            cha_inv = _fdsn_client_get_inventory(
                client, ch_row, response_level="response", max_tries=10
            )

            for returned_chan in cha_inv.networks[0].stations[0].channels:
                station_obj.channels.append(returned_chan)

            # -----------------------------
            # get data if desired
            if data:
                streams += self.get_waveforms_from_request_row(client, ch_row)

        # Pack the stations into networks
        for network_key in stations_dict.keys():
            for start_key in stations_dict[network_key].keys():
                for station_id, packed_station in stations_dict[network_key][
                    start_key
                ].items():
                    networks_dict[network_key][start_key].stations.append(
                        packed_station
                    )
        # Pack the networks into the inventory
        for network_key in networks_dict.keys():
            for start_key in networks_dict[network_key].keys():
                inv.networks.append(networks_dict[network_key][start_key])
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
        return pd.DataFrame(rows, columns=self.request_columns)

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
        networks = df["network"].unique()
        for network in networks:
            network_dict = {
                "network": network,
                "stations": df[df.network == network].station.unique().tolist(),
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
            "_".join([f"{d['network']}_{'_'.join(d['stations'])}" for d in unique_list])
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


def _fdsn_client_get_inventory(client, row, response_level, max_tries=10):
    """
    Allows a few tries to get inventory, in case server is not very responsive
    Parameters
    ----------
    client: obspy.clients.fdsn.Client
        obspy helper to get data from FDSN (e.g. EarthScope)
    row: pandas.core.frame.Pandas
        A row of a dataframe specifying the start and end times, station and network
    response_level: ["network", "station", "response"]

    Returns
    -------

    # TODO: Maybe these two cases can be the same call to client.get_stations?

    """
    from lxml.etree import XMLSyntaxError

    def sleep_random_time():
        """Sleep for a fraction of a second before trying again"""
        sleep_time = np.random.randint(0, 100) * 0.01
        logger.info(f"Sleeping for {sleep_time}s")
        time.sleep(sleep_time)
        return

    i_try = 0
    if response_level == "station":
        while i_try < max_tries:
            try:
                inventory = client.get_stations(
                    row.start,
                    row.end,
                    network=row.network,
                    station=row.station,
                    level=response_level,
                )
                i_try += max_tries
            except (BadGzipFile, XMLSyntaxError, ValueError) as e:
                logger.error(f"{e}")
                msg = f"Failed to get Station {row.network}-{row.station} inventory try {i_try} of {max_tries}"
                logger.warning(msg)
                sleep_random_time()
                i_try += 1

    if response_level == "response":  # channel level
        while i_try < max_tries:
            try:
                inventory = client.get_stations(
                    row.start,
                    row.end,
                    network=row.network,
                    station=row.station,
                    loc=row.location,
                    channel=row.channel,
                    level=response_level,
                )
                i_try += max_tries
            except (BadGzipFile, XMLSyntaxError, ValueError) as e:
                logger.error(f"{e}")
                msg = f"Failed to get Channel {row.network}-{row.station}-{row.channel} inventory try {i_try} of {max_tries}"
                logger.warning(msg)
                sleep_random_time()
                i_try += 1

    if response_level == "network":
        try:
            inventory = client.get_stations(
                row.start,
                row.end,
                network=row.network,
                level=response_level,
            )
            i_try += max_tries
        except (BadGzipFile, XMLSyntaxError, ValueError) as e:
            logger.error(f"{e}")
            msg = f"Failed to get Network {row.network}-{row.station}-{row.channel} inventory try {i_try} of {max_tries}"
            logger.warning(msg)
            sleep_random_time()
            i_try += 1

    return inventory
