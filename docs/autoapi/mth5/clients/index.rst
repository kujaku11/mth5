mth5.clients
============

.. py:module:: mth5.clients


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/clients/base/index
   /autoapi/mth5/clients/fdsn/index
   /autoapi/mth5/clients/geomag/index
   /autoapi/mth5/clients/lemi424/index
   /autoapi/mth5/clients/make_mth5/index
   /autoapi/mth5/clients/metronix/index
   /autoapi/mth5/clients/nims/index
   /autoapi/mth5/clients/phoenix/index
   /autoapi/mth5/clients/zen/index


Classes
-------

.. autoapisummary::

   mth5.clients.FDSN
   mth5.clients.USGSGeomag
   mth5.clients.PhoenixClient
   mth5.clients.ZenClient
   mth5.clients.LEMI424Client
   mth5.clients.MetronixClient
   mth5.clients.NIMSClient


Package Contents
----------------

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



.. py:class:: USGSGeomag(**kwargs)

   .. py:attribute:: save_path


   .. py:attribute:: mth5_filename
      :value: None



   .. py:attribute:: interact
      :value: False



   .. py:attribute:: request_columns
      :value: ['observatory', 'type', 'elements', 'sampling_period', 'start', 'end']



   .. py:attribute:: h5_compression
      :value: 'gzip'



   .. py:attribute:: h5_compression_opts
      :value: 4



   .. py:attribute:: h5_shuffle
      :value: True



   .. py:attribute:: h5_fletcher32
      :value: True



   .. py:attribute:: h5_data_level
      :value: 1



   .. py:attribute:: mth5_file_mode
      :value: 'w'



   .. py:attribute:: mth5_version
      :value: '0.2.0'



   .. py:property:: h5_kwargs


   .. py:method:: validate_request_df(request_df)

      Make sure the input request dataframe has the appropriate columns

      :param request_df: request dataframe
      :type request_df: :class:`pandas.DataFrame`
      :return: valid request dataframe
      :rtype: :class:`pandas.DataFrame`




   .. py:method:: add_run_id(request_df)

      Add run id to request df

      :param request_df: request dataframe
      :type request_df: :class:`pandas.DataFrame`
      :return: add a run number to unique time windows for each observatory
       at each unique sampling period.
      :rtype: :class:`pandas.DataFrame`




   .. py:method:: make_mth5_from_geomag(request_df)

      Download geomagnetic observatory data from USGS webservices into an
      MTH5 using a request dataframe or csv file.

      :param request_df: DataFrame with columns

          - 'observatory'     --> Observatory code
          - 'type'            --> data type [ 'variation' | 'adjusted' | 'quasi-definitive' | 'definitive' ]
          - 'elements'        --> Elements to get [D, DIST, DST, E, E-E, E-N, F, G, H, SQ, SV, UK1, UK2, UK3, UK4, X, Y, Z]
          - 'sampling_period' --> sample period [ 1 | 60 | 3600 ]
          - 'start'           --> Start time YYYY-MM-DDThh:mm:ss
          - 'end'             --> End time YYYY-MM-DDThh:mm:ss

      :type request_df: :class:`pandas.DataFrame`, str or Path if csv file


      :return: if interact is True an MTH5 object is returned otherwise the
       path to the file is returned
      :rtype: Path or :class:`mth5.mth5.MTH5`

      .. seealso:: https://www.usgs.gov/tools/web-service-geomagnetism-data




.. py:class:: PhoenixClient(data_path: str | pathlib.Path, sample_rates: list[int] = [150, 24000], save_path: str | pathlib.Path | None = None, receiver_calibration_dict: dict | str | pathlib.Path = {}, sensor_calibration_dict: dict | str | pathlib.Path = {}, mth5_filename: str = 'from_phoenix.h5', **kwargs: dict)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:property:: receiver_calibration_dict
      :type: dict


      Receiver calibrations.

      :returns: Dictionary mapping receiver IDs to calibration file paths.
      :rtype: dict

      .. rubric:: Examples

      >>> client = PhoenixClient('data/path')
      >>> client.receiver_calibration_dict = {'RX001': Path('RX001_rxcal.json')}
      >>> client.receiver_calibration_dict
      {'RX001': Path('RX001_rxcal.json')}


   .. py:property:: sensor_calibration_dict
      :type: dict


      Sensor calibration dictionary.

      :returns: Dictionary mapping sensor IDs to PhoenixCalibration objects.
      :rtype: dict

      .. rubric:: Examples

      >>> client = PhoenixClient('data/path')
      >>> client.sensor_calibration_dict = {'H001': PhoenixCalibration('H001_scal.json')}
      >>> client.sensor_calibration_dict['H001']  # doctest: +SKIP
      <PhoenixCalibration object>


   .. py:attribute:: collection


   .. py:method:: make_mth5_from_phoenix(**kwargs: dict) -> str | pathlib.Path | None

      Make an MTH5 from Phoenix files.

      Split into runs, account for filters. Updates the MTH5 file with Phoenix data.

      :param \*\*kwargs: Optional keyword arguments to override instance attributes.
      :type \*\*kwargs: dict

      :returns: Path to the saved MTH5 file.
      :rtype: str, Path, or None

      .. rubric:: Examples

      >>> client = PhoenixClient('data/path', save_path='output.h5')
      >>> client.make_mth5_from_phoenix()
      'output.h5'



.. py:class:: ZenClient(data_path, sample_rates=[4096, 1024, 256], save_path=None, calibration_path=None, mth5_filename='from_zen.h5', **kwargs)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:property:: calibration_path

      Path to calibration data


   .. py:attribute:: collection


   .. py:attribute:: station_stem
      :value: None



   .. py:method:: get_run_dict()

      Get Run information

      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_survey(station_dict)

      get survey name from a dictionary of a single station of runs
      :param station_dict: DESCRIPTION
      :type station_dict: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: make_mth5_from_zen(survey_id=None, combine=True, **kwargs)

      Make an MTH5 from Phoenix files.  Split into runs, account for filters

      :param data_path: DESCRIPTION, defaults to None
      :type data_path: TYPE, optional
      :param sample_rates: DESCRIPTION, defaults to None
      :type sample_rates: TYPE, optional
      :param save_path: DESCRIPTION, defaults to None
      :type save_path: TYPE, optional
      :return: DESCRIPTION
      :rtype: TYPE




.. py:class:: LEMI424Client(data_path: Union[str, pathlib.Path], save_path: Optional[Union[str, pathlib.Path]] = None, mth5_filename: str = 'from_lemi424.h5', **kwargs: Any)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:attribute:: collection


   .. py:method:: make_mth5_from_lemi424(survey_id: str, station_id: str, **kwargs: Any) -> pathlib.Path

      Create an MTH5 file from LEMI 424 long period data.

      :param survey_id: Survey identifier.
      :type survey_id: str
      :param station_id: Station identifier.
      :type station_id: str
      :param \*\*kwargs: Additional keyword arguments to set as attributes.
      :type \*\*kwargs: Any

      :returns: Path to the created mth5 file.
      :rtype: Path

      .. rubric:: Examples

      >>> client = LEMI424Client(data_path="./data")
      >>> client.make_mth5_from_lemi424("SURVEY1", "ST01")
      PosixPath('data/from_lemi424.h5')



.. py:class:: MetronixClient(data_path, sample_rates=[128], save_path=None, calibration_path=None, mth5_filename='from_metronix.h5', **kwargs)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:attribute:: calibration_path
      :value: None



   .. py:attribute:: collection


   .. py:method:: get_run_dict(run_name_zeros=0)

      get run information

      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_survey_id(station_dict)

      get survey name from a dictionary of a single station of runs
      :param station_dict: DESCRIPTION
      :type station_dict: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: set_station_metadata(station_dict, station_group)

      set station group metadata from information in the station dict

      :param station_dict: DESCRIPTION
      :type station_dict: TYPE
      :param station_group: DESCRIPTION
      :type station_group: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: make_mth5_from_metronix(run_name_zeros=0, **kwargs)

      Create an MTH5 from new ATSS + JSON style Metronix data.

      :param **kwargs: DESCRIPTION
      :type **kwargs: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




.. py:class:: NIMSClient(data_path: str | pathlib.Path, sample_rates: list[int] = [1, 8], save_path: str | pathlib.Path | None = None, calibration_path: str | pathlib.Path | None = None, mth5_filename: str = 'from_nims.h5', **kwargs)

   Bases: :py:obj:`mth5.clients.base.ClientBase`


   .. py:property:: calibration_path
      :type: pathlib.Path | None


      Path to calibration data.

      :returns: Path to calibration file, or None if not set.
      :rtype: Path or None

      .. rubric:: Examples

      >>> client = NIMSClient('data_dir')
      >>> client.calibration_path = 'calib.dat'
      >>> print(client.calibration_path)
      PosixPath('calib.dat')


   .. py:attribute:: collection


   .. py:method:: get_run_dict() -> dict

      Get run information from the NIMS collection.

      :returns: Dictionary of run information.
      :rtype: dict

      .. rubric:: Examples

      >>> client = NIMSClient('data_dir')
      >>> runs = client.get_run_dict()
      >>> print(list(runs.keys()))
      ['station1', 'station2']



   .. py:method:: get_survey(station_dict: dict) -> str

      Get survey name from a dictionary of a single station of runs.

      :param station_dict: Dictionary of runs for a station.
      :type station_dict: dict

      :returns: Survey name.
      :rtype: str

      .. rubric:: Examples

      >>> client = NIMSClient('data_dir')
      >>> runs = client.get_run_dict()
      >>> survey = client.get_survey(runs['station1'])
      >>> print(survey)
      'survey_name'



   .. py:method:: make_mth5_from_nims(survey_id: str = 'default_survey', combine: bool = True, **kwargs) -> str | pathlib.Path

      Make an MTH5 file from Phoenix NIMS files. Splits into runs, accounts for filters.

      :param survey_id: Survey identifier. Default is "default_survey".
      :type survey_id: str, optional
      :param combine: Whether to combine runs. Default is True.
      :type combine: bool, optional
      :param \*\*kwargs: Additional keyword arguments to set as attributes.

      :returns: Path to the saved MTH5 file.
      :rtype: str or Path

      .. rubric:: Examples

      >>> client = NIMSClient('data_dir')
      >>> mth5_path = client.make_mth5_from_nims(survey_id='survey1')
      >>> print(mth5_path)
      'output_dir/from_nims.h5'



