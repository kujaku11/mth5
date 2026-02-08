mth5.clients.fdsn
=================

.. py:module:: mth5.clients.fdsn

.. autoapi-nested-parse::

   FDSN
   =========

   Module for working with FDSN clients using Obspy

   Created on Fri Feb  4 15:53:21 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.clients.fdsn.FDSN


Module Contents
---------------

.. py:class:: FDSN(client: str = 'IRIS', **kwargs)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:attribute:: logger


   .. py:attribute:: request_columns
      :value: ['network', 'station', 'location', 'channel', 'start', 'end']



   .. py:attribute:: client
      :value: 'IRIS'



   .. py:property:: run_list_ne_stream_intervals_message
      :type: str


      note about not equal stream intervals


   .. py:method:: get_run_list_from_station_id(m: mth5.mth5.MTH5, station_id: str, survey_id: str | None = None) -> list[str]

      ignored_groups created to address issue #153.  This might be better placed
      closer to the core of mth5.

      :param m:
      :param station_id:

      :returns: **run_list**
      :rtype: list of strings



   .. py:method:: stream_boundaries(streams: obspy.Stream) -> tuple[list[obspy.UTCDateTime], list[obspy.UTCDateTime]]

      Identify start and end times of streams

      :param streams:
      :type streams: obspy.core.stream.Stream



   .. py:method:: get_station_streams(station_id: str) -> obspy.Stream

      Get streams for a certain station



   .. py:method:: get_run_group(mth5_obj_or_survey, station_id: str, run_id: str)

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
      :param mth5_obj_or_survey:
      :type mth5_obj_or_survey: mth5.mth5.MTH5 or mth5.groups.survey.SurveyGroup



   .. py:method:: pack_stream_into_run_group(run_group, run_stream: obspy.Stream)


   .. py:method:: run_timings_match_stream_timing(run_group, stream_start: obspy.UTCDateTime, stream_end: obspy.UTCDateTime) -> bool

      Checks start and end times in the run.
      Compares start and end times of runs to start and end times of traces.
      If True, will packs runs based on time spans.

      :param run_group:
      :type run_group: mth5.groups.run.RunGroup
      :param stream_start:
      :type stream_start: obspy.UTCDateTime
      :param stream_end:
      :type stream_end: obspy.UTCDateTime

      :rtype: bool



   .. py:method:: wrangle_runs_into_containers(m: mth5.mth5.MTH5, station_id: str, survey_group=None) -> None

      Note 1: There used to be two separate functions for this, but now there
      is one run_group_source is defined as either m or survey_group depending
      on v0.1.0 or 0.2.0

      Note 2: If/elif/elif/else Logic:
      The strategy is to add the group first. This will get the already filled
      in metadata to update the run_ts_obj. Then get streams an add existing
      metadata.


      :param m:
      :param streams:
      :param station_id:
      :param survey_group:



   .. py:method:: make_mth5_from_fdsn_client(df: pandas.DataFrame | str | pathlib.Path, path: str | pathlib.Path | None = None, client: str | None = None, interact: bool = False) -> pathlib.Path

      Create an MTH5 file from an FDSN data center request.

      :param df:
                 DataFrame or path to CSV with columns:
                     - 'network'   : FDSN Network code
                     - 'station'   : FDSN Station code
                     - 'location'  : FDSN Location code
                     - 'channel'   : FDSN Channel code
                     - 'start'     : Start time YYYY-MM-DDThh:mm:ss
                     - 'end'       : End time YYYY-MM-DDThh:mm:ss
      :type df: pandas.DataFrame or str or Path
      :param path: Path to save MTH5 file (default: current directory).
      :type path: str or Path, optional
      :param client: FDSN client name (default: "IRIS").
      :type client: str, optional
      :param interact: Deprecated. If True, logs a warning (default: False).
      :type interact: bool, optional

      :returns: **file_name** -- Path to the created MTH5 file.
      :rtype: Path

      :raises AttributeError: If the input DataFrame is not properly formatted.
      :raises ValueError: If the values of the DataFrame are not correct.

      .. rubric:: Examples

      >>> from mth5.clients.fdsn import FDSN
      >>> import pandas as pd
      >>> df = pd.DataFrame({
      ...     'network': ['XX'],
      ...     'station': ['1234'],
      ...     'location': [''],
      ...     'channel': ['LHZ'],
      ...     'start': ['2022-01-01T00:00:00'],
      ...     'end': ['2022-01-02T00:00:00']
      ... })
      >>> client = FDSN()
      >>> file_path = client.make_mth5_from_fdsn_client(df)



   .. py:property:: streams

      obspy.Stream object


   .. py:method:: make_mth5_from_inventory_and_streams(inventory: obspy.Inventory | str | pathlib.Path, streams: obspy.Stream | list[str | pathlib.Path], save_path: str | pathlib.Path | None = None) -> pathlib.Path

      Create an MTH5 file from an ObsPy Inventory and waveform streams.

      :param inventory: ObsPy Inventory object or path to StationXML file.
      :type inventory: obspy.Inventory or str or Path
      :param streams: ObsPy Stream object or list of file paths to waveform data.
      :type streams: obspy.Stream or list of str or Path
      :param save_path: Path to save MTH5 file (default: current directory).
      :type save_path: str or Path, optional

      :returns: **file_name** -- Path to the created MTH5 file.
      :rtype: Path

      .. rubric:: Examples

      >>> from mth5.clients.fdsn import FDSN
      >>> inv = ... # ObsPy Inventory
      >>> streams = ... # ObsPy Stream
      >>> client = FDSN()
      >>> file_path = client.make_mth5_from_inventory_and_streams(inv, streams)



   .. py:method:: build_network_dict(df: pandas.DataFrame, client: obspy.clients.fdsn.Client) -> dict

      Build a dictionary of networks keyed by network_id and start_time.

      :param df: Request DataFrame.
      :type df: pandas.DataFrame
      :param client: FDSN client instance.
      :type client: obspy.clients.fdsn.Client

      :returns: **networks** -- Dictionary of networks.
      :rtype: dict

      .. rubric:: Examples

      >>> networks = client.build_network_dict(df, client)



   .. py:method:: build_station_dict(df: pandas.DataFrame, client: obspy.clients.fdsn.Client, networks_dict: dict) -> dict

      Build a dictionary of stations keyed by network_id and start_time.

      :param df: Request DataFrame.
      :type df: pandas.DataFrame
      :param client: FDSN client instance.
      :type client: obspy.clients.fdsn.Client
      :param networks_dict: Dictionary of networks.
      :type networks_dict: dict

      :returns: **stations** -- Dictionary of stations.
      :rtype: dict

      .. rubric:: Examples

      >>> stations = client.build_station_dict(df, client, networks_dict)



   .. py:method:: get_waveforms_from_request_row(client: obspy.clients.fdsn.Client, row) -> obspy.Stream

      Retrieve waveform data for a request row.

      :param client: FDSN client instance.
      :type client: obspy.clients.fdsn.Client
      :param row: Row of request DataFrame.
      :type row: pandas.Series

      :returns: **streams** -- ObsPy Stream object with waveform data.
      :rtype: obspy.Stream

      .. rubric:: Examples

      >>> streams = client.get_waveforms_from_request_row(client, row)



   .. py:method:: get_inventory_from_df(df: pandas.DataFrame | str | pathlib.Path, client: str | None = None, data: bool = True, max_tries: int = 10) -> tuple[obspy.Inventory, obspy.Stream]

      Get an ObsPy Inventory and Stream from a DataFrame request.

      :param df:
                 DataFrame or path to CSV with columns:
                     - 'network'   : FDSN Network code
                     - 'station'   : FDSN Station code
                     - 'location'  : FDSN Location code
                     - 'channel'   : FDSN Channel code
                     - 'start'     : Start time YYYY-MM-DDThh:mm:ss
                     - 'end'       : End time YYYY-MM-DDThh:mm:ss
      :type df: pandas.DataFrame or str or Path
      :param client: FDSN client name (default: self.client).
      :type client: str, optional
      :param data: If True, retrieves waveform data (default: True).
      :type data: bool, optional
      :param max_tries: Maximum number of retry attempts (default: 10).
      :type max_tries: int, optional

      :returns: * **inventory** (*obspy.Inventory*) -- Inventory of metadata requested.
                * **streams** (*obspy.Stream*) -- Stream of waveform data.

      .. rubric:: Examples

      >>> from mth5.clients.fdsn import FDSN
      >>> import pandas as pd
      >>> df = pd.DataFrame({
      ...     'network': ['XX'],
      ...     'station': ['1234'],
      ...     'location': [''],
      ...     'channel': ['LHZ'],
      ...     'start': ['2022-01-01T00:00:00'],
      ...     'end': ['2022-01-02T00:00:00']
      ... })
      >>> client = FDSN()
      >>> inv, streams = client.get_inventory_from_df(df)



   .. py:method:: get_df_from_inventory(inventory: obspy.Inventory) -> pandas.DataFrame

      Create a DataFrame from an ObsPy Inventory object.

      :param inventory: ObsPy Inventory object.
      :type inventory: obspy.Inventory

      :returns: **df** -- DataFrame in request format.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      >>> df = client.get_df_from_inventory(inventory)



   .. py:method:: get_unique_networks_and_stations(df: pandas.DataFrame) -> list[dict]

      Get unique networks and stations from a request DataFrame.

      :param df: Request DataFrame.
      :type df: pandas.DataFrame

      :returns: **unique_list** -- List of network dictionaries with stations.
      :rtype: list of dict

      .. rubric:: Examples

      >>> unique_list = client.get_unique_networks_and_stations(df)



   .. py:method:: make_filename(df: pandas.DataFrame) -> str

      Make a filename from a request DataFrame of networks and stations.

      :param df: Request DataFrame.
      :type df: pandas.DataFrame

      :returns: **filename** -- Filename in the format network_01+stations_network_02+stations.h5
      :rtype: str

      .. rubric:: Examples

      >>> filename = client.make_filename(df)



   .. py:method:: get_fdsn_channel_map() -> dict[str, str]

      Get mapping of FDSN channel codes to internal codes.

      :returns: **FDSN_CHANNEL_MAP** -- Dictionary mapping FDSN channel codes.
      :rtype: dict

      .. rubric:: Examples

      >>> channel_map = client.get_fdsn_channel_map()



