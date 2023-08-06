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
from pathlib import Path

import pandas as pd

from obspy.clients import fdsn
from obspy import UTCDateTime
from obspy import read as obsread
from obspy.core.inventory import Inventory

from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment

from mth5.mth5 import MTH5
from mth5.timeseries import RunTS


class FDSN:
    def __init__(self, client="IRIS", mth5_version="0.2.0", **kwargs):
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
        self.compression = "gzip"
        self.compression_opts = 4
        self.shuffle = True
        self.fletcher32 = True
        self.data_level = 1
        self.mth5_version = mth5_version

        for key, value in kwargs.items():
            setattr(self, key, value)

        # ivars
        self._streams = None

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
                f"column names in file {df.columns} are not the expected {self.request_columns}"
            )
        return df

    @property
    def run_list_ne_stream_intervals_message(self):
        print(
            "More or less runs have been requested by the user "
            + "than are defined in the metadata. Runs will be "
            + "defined but only the requested run extents contain "
            + "time series data "
            + "based on the users request."
        )

    def _loop_stations(self, stations, m, survey_group=None):
        for station_id in stations:
            self.wrangle_runs_into_containers(m, station_id, survey_group=survey_group)

    def _run_010(self, unique_list, m, **kwargs):
        """
        kwargs are supported just to make this a general function that can be kept in a dict
        and used as in process_list

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
        Routs job to correct processing based on mth5_version
        Maintainable way to handle future file versions and send them to their own processing functions if needed
        Parameters
        ----------
        experiment
        unique_list
        m

        Returns
        -------

        """

        version_dict = {"0.1.0": self._run_010,
                       "0.2.0": self._run_020}

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
        start_times = [UTCDateTime(x) for x in start_times]
        end_times = [UTCDateTime(x) for x in end_times]
        return start_times, end_times

    def get_station_streams(self, station_id):
        return self._streams.select(station=station_id)

    def get_run_group(self, mth5_obj_or_survey, station_id, run_id):
        """
        This method is key to merging wrangle_runs_into_containers_v1 and
        wrangle_runs_into_containers_v2.
        Because a v1 mth5 object can get a survey group with the same method as can a v2 survey_group

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
        return

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
        cond1 = stream_start >= UTCDateTime(run_start)
        cond2 = stream_end <= UTCDateTime(run_end)
        if cond1 and cond2:  # paired up
            streams_and_run_timings_match = True
        return streams_and_run_timings_match

    def wrangle_runs_into_containers(self, m, station_id, survey_group=None):
        """
        Note 1: There used to be two separate functions for this, but now there is one
        run_group_source is defined as either m or survey_group depending on v0.1.0
        or 0.2.0

        Note 2: If/elif/elif/else Logic:
        The strategy is to add the group first. This will get the already filled in
        metadata to update the run_ts_obj. Then get streams an add existing metadata.


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
        if len(run_list) == num_streams:
            for run_id, start, end in zip(run_list, trace_start_times, trace_end_times):
                run_group = self.get_run_group(run_group_source, station_id, run_id)
                run_stream = msstreams.slice(start, end)
                self.pack_stream_into_run_group(run_group, run_stream)
        elif len(run_list) == 1:
            for run_id, times in enumerate(zip(trace_start_times, trace_end_times), 1):
                start = times[0]
                end = times[1]
                run_id = f"{run_id:03}"
                run_group = self.get_run_group(run_group_source, station_id, run_id)
                run_stream = msstreams.slice(start, end)
                self.pack_stream_into_run_group(run_group, run_stream)
        elif len(run_list) != num_streams:
            self.run_list_ne_stream_intervals_message
            for run_id, start, end in zip(run_list, trace_start_times, trace_end_times):
                for run in run_list:
                    run_group = self.get_run_group(run_group_source, station_id, run)
                    if self.run_timings_match_stream_timing(run_group, start, end):
                        run_stream = msstreams.slice(start, end)
                        self.pack_stream_into_run_group(run_group, run_stream)
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
        if path is None:
            path = Path().cwd()
        else:
            path = Path(path)
        if client is not None:
            self.client = client
        df = self._validate_dataframe(df)

        unique_list = self.get_unique_networks_and_stations(df)
        if self.mth5_version in ["0.1.0"]:
            if len(unique_list) != 1:
                raise AttributeError("MTH5 supports one survey/network per container.")
        file_name = path.joinpath(self.make_filename(df))

        # initiate MTH5 file
        m = MTH5(
            file_version=self.mth5_version,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
            data_level=self.data_level,
        )
        m.open_mth5(file_name, "w")

        # read in inventory and streams
        inv, streams = self.get_inventory_from_df(df, self.client)
        self._streams = streams

        # translate obspy.core.Inventory to an mt_metadata.timeseries.Experiment
        translator = XMLInventoryMTExperiment()
        experiment = translator.xml_to_mt(inv)

        # Updates expriment information based on time extent of streams
        # rather than time extent of inventory
        # experiment = translator.drop_runs(m, streams)

        m.from_experiment(experiment)
        self._process_list(experiment, unique_list, m)

        if interact:
            return m
        else:
            m.close_mth5()
            return file_name


    def get_waveforms_from_request_row(self, client, row):
        """

        Parameters
        ----------
        row

        Returns
        -------

        """
        start = UTCDateTime(row.start)
        end = UTCDateTime(row.end)
        streams = client.get_waveforms(row.network, row.station, row.location, row.channel, start, end)
        return streams

    def get_inventory_from_df(self, df, client=None, data=True):
        """
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
        client = fdsn.Client(self.client)

        # creat an empty stream to add to
        streams = obsread()
        streams.clear()

        inv = Inventory(networks=[], source="MTH5")

        # sort the values to be logically ordered
        df.sort_values(self.request_columns[:-1])

        used_network = dict()
        used_station = dict()
        for row in df.itertuples():
            # First for loop builds out networks and stations
            if row.network not in used_network:
                net_inv = client.get_stations(row.start, row.end, network=row.network, level="network")
                returned_network = net_inv.networks[0]
                used_network[row.network] = [row.start]
            elif used_network.get(row.network) is not None and row.start not in used_network.get(row.network):
                net_inv = client.get_stations(row.start, row.end, network=row.network, level="network")
                returned_network = net_inv.networks[0]
                used_network[row.network].append(row.start)
            else:
                continue
            for st_row in df.itertuples():
                if row.network != st_row.network:
                    continue
                else:
                    if st_row.station not in used_station:
                        sta_inv = client.get_stations(
                            st_row.start,
                            st_row.end,
                            network=row.network,
                            station=st_row.station,
                            level="station",
                        )
                        returned_sta = sta_inv.networks[0].stations[0]
                        used_station[st_row.station] = [st_row.start]
                    elif used_station.get(
                        st_row.station
                    ) is not None and st_row.start not in used_station.get(
                        st_row.station
                    ):
                        # Checks for epoch
                        sta_inv = client.get_stations(
                            st_row.start,
                            st_row.end,
                            network=st_row.network,
                            station=st_row.station,
                            level="station",
                        )
                        returned_sta = sta_inv.networks[0].stations[0]
                        used_station[st_row.station].append(st_row.start)
                    else:
                        continue
                for ch_row in df.itertuples():
                    if (
                        ch_row.network == row.network
                        and st_row.station == ch_row.station
                        and ch_row.start == st_row.start
                    ):
                        cha_inv = client.get_stations(
                            ch_row.start,
                            ch_row.end,
                            network=ch_row.network,
                            station=ch_row.station,
                            loc=ch_row.location,
                            channel=ch_row.channel,
                            level="response",
                        )
                        for returned_chan in cha_inv.networks[0].stations[0].channels:
                            returned_sta.channels.append(returned_chan)

                        # -----------------------------
                        # get data if desired
                        if data:
                            streams += self.get_waveforms_from_request_row(client, ch_row)
                    else:
                        continue
                returned_network.stations.append(returned_sta)
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
