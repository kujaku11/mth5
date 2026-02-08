mth5.clients.make_mth5
======================

.. py:module:: mth5.clients.make_mth5

.. autoapi-nested-parse::

   Make MTH5
   ============

   This module provides helper functions to make MTH5 file from various clients

   Supported Clients include:

       * FDSN (through Obspy)
       * Science Base (TODO)
       * NCI - Australia (TODO)
       * Phoenix MTU-5C
       * Zen
       * LEMI 424
       * Phoenix legacy MTU (TODO)
       * Metronix Geophysics
       * USGS Geomagnetism


   Updated on Wed Aug  25 19:57:00 2021

   @author: jpeacock + tronan



Classes
-------

.. autoapisummary::

   mth5.clients.make_mth5.MakeMTH5


Module Contents
---------------

.. py:class:: MakeMTH5(mth5_version: str = '0.2.0', interact: bool = False, save_path: str | pathlib.Path | None = None, **kwargs)

   Factory class for creating MTH5 files from various data sources.

   This class provides class methods to create MTH5 files from different
   magnetotelluric data acquisition systems and data repositories.

   :param mth5_version: MTH5 file format version
   :type mth5_version: str, default "0.2.0"
   :param interact: If True, keep file open for interactive use. If False, close file after creation.
   :type interact: bool, default False
   :param save_path: Directory path to save MTH5 file. If None, uses current working directory.
   :type save_path: str or Path, optional
   :param \*\*kwargs: Additional keyword arguments for HDF5 file parameters. Any parameter
                      starting with 'h5' will be used for HDF5 configuration.
   :type \*\*kwargs: dict

   .. attribute:: h5_compression

      HDF5 compression algorithm

      :type: str, default "gzip"

   .. attribute:: h5_compression_opts

      Compression level (0-9 for gzip)

      :type: int, default 4

   .. attribute:: h5_shuffle

      Enable byte shuffle filter for better compression

      :type: bool, default True

   .. attribute:: h5_fletcher32

      Enable Fletcher32 checksum for data integrity

      :type: bool, default True

   .. attribute:: h5_data_level

      Data processing level indicator

      :type: int, default 1

   .. rubric:: Examples

   Create a basic MakeMTH5 instance:

   >>> from mth5.clients import MakeMTH5
   >>> maker = MakeMTH5(save_path="/path/to/save")
   >>> print(maker)
   MakeMTH5 Attibutes:
       mth5_version: 0.2.0
       h5_compression: gzip
       ...

   Create with custom compression:

   >>> maker = MakeMTH5(
   ...     save_path="/data/mt",
   ...     h5_compression="lzf",
   ...     h5_shuffle=False
   ... )

   .. seealso::

      :obj:`mth5.mth5.MTH5`
          Main MTH5 file interface


   .. py:attribute:: mth5_version
      :value: '0.2.0'



   .. py:attribute:: interact
      :value: False



   .. py:property:: save_path
      :type: pathlib.Path


      Get the save path as a Path object.


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



   .. py:attribute:: mth5_filename
      :value: 'make_mth5.h5'



   .. py:method:: get_h5_kwargs() -> dict

      Extract HDF5-related keyword arguments from instance attributes.

      :returns: Dictionary of HDF5 configuration parameters including version,
                compression settings, shuffle, fletcher32, and data level.
      :rtype: dict

      .. rubric:: Examples

      >>> maker = MakeMTH5(h5_compression="lzf", h5_data_level=2)
      >>> kwargs = maker.get_h5_kwargs()
      >>> print(kwargs["h5_compression"])
      lzf



   .. py:method:: from_fdsn_client(request_df: pandas.DataFrame, client: str = 'IRIS', **kwargs)
      :classmethod:


      Create MTH5 file from FDSN data service.

      Pull data from an FDSN archive like IRIS using ObsPy clients.
      The request DataFrame specifies which data to download.

      :param request_df: DataFrame with columns:

                         - 'network' : str
                             FDSN Network code (e.g., 'IU', 'TA')
                         - 'station' : str
                             FDSN Station code (e.g., 'ANMO', 'CAS04')
                         - 'location' : str
                             FDSN Location code (e.g., '00', '')
                         - 'channel' : str
                             FDSN Channel code (e.g., 'LFE', 'BHZ')
                         - 'start' : str
                             Start time in format 'YYYY-MM-DDThh:mm:ss'
                         - 'end' : str
                             End time in format 'YYYY-MM-DDThh:mm:ss'
      :type request_df: pd.DataFrame
      :param client: FDSN client name (e.g., 'IRIS', 'USGS', 'NCEDC')
      :type client: str, default "IRIS"
      :param \*\*kwargs: Additional keyword arguments. HDF5 parameters should be prefixed
                         with 'h5_' (e.g., h5_compression='gzip', h5_compression_opts=4).
      :type \*\*kwargs: dict

      :returns: MTH5 object if interact=True, otherwise Path to created file
      :rtype: mth5.mth5.MTH5 or Path

      :raises AttributeError: If the input DataFrame is not properly formatted
      :raises ValueError: If the DataFrame column values are invalid

      .. rubric:: Notes

      If any column value is blank, any matching value will be searched.
      For example, leaving 'station' blank will return all stations within
      the specified time range.

      .. rubric:: Examples

      Create a request DataFrame and download data:

      >>> import pandas as pd
      >>> from mth5.clients import MakeMTH5
      >>>
      >>> # Define request
      >>> request_df = pd.DataFrame({
      ...     'network': ['IU'],
      ...     'station': ['ANMO'],
      ...     'location': ['00'],
      ...     'channel': ['LF*'],
      ...     'start': ['2020-01-01T00:00:00'],
      ...     'end': ['2020-01-02T00:00:00']
      ... })
      >>>
      >>> # Create MTH5 with custom compression
      >>> mth5_obj = MakeMTH5.from_fdsn_client(
      ...     request_df,
      ...     client='IRIS',
      ...     h5_compression_opts=1
      ... )

      .. seealso::

         :obj:`from_fdsn_miniseed_and_stationxml`
             Create from existing files

         :obj:`obspy.clients.fdsn`
             ObsPy FDSN client documentation



   .. py:method:: from_fdsn_miniseed_and_stationxml(station_xml_path: str | pathlib.Path, miniseed_files: str | pathlib.Path | list[str | pathlib.Path], save_path: str | pathlib.Path | None = None, **kwargs)
      :classmethod:


      Create MTH5 from existing StationXML and miniSEED files.

      Use this method when you already have StationXML and miniSEED files
      downloaded from an FDSN client or created locally.

      :param station_xml_path: Full path to StationXML file or an ObsPy Inventory object
      :type station_xml_path: str, Path, or obspy.Inventory
      :param miniseed_files: List of miniSEED file paths or ObsPy Stream objects. Can also
                             be a single file path or Stream object.
      :type miniseed_files: str, Path, list, or obspy.Stream
      :param save_path: Directory to save new MTH5 file. If None, saves to current
                        working directory.
      :type save_path: str or Path, optional
      :param \*\*kwargs: Additional keyword arguments. HDF5 parameters should be prefixed
                         with 'h5_' (e.g., h5_compression='gzip').
      :type \*\*kwargs: dict

      :returns: Path to created MTH5 file. Filename format is
                {network}_{station}.h5 based on unique network and station codes.
      :rtype: Path

      :raises TypeError: If inputs are not of correct type

      .. rubric:: Examples

      Create MTH5 from existing files:

      >>> from mth5.clients import MakeMTH5
      >>> from pathlib import Path
      >>>
      >>> # Define file paths
      >>> station_xml = Path("data/station.xml")
      >>> miniseed = [
      ...     Path("data/IU.ANMO.00.LFE.mseed"),
      ...     Path("data/IU.ANMO.00.LFN.mseed")
      ... ]
      >>>
      >>> # Create MTH5
      >>> mth5_path = MakeMTH5.from_fdsn_miniseed_and_stationxml(
      ...     station_xml,
      ...     miniseed,
      ...     save_path="output",
      ...     h5_compression="lzf"
      ... )
      >>> print(mth5_path)
      output/IU_ANMO.h5

      Using ObsPy objects directly:

      >>> from obspy import read, read_inventory
      >>>
      >>> inventory = read_inventory("station.xml")
      >>> stream = read("data/*.mseed")
      >>>
      >>> mth5_path = MakeMTH5.from_fdsn_miniseed_and_stationxml(
      ...     inventory,
      ...     stream,
      ...     save_path="output"
      ... )

      .. seealso::

         :obj:`from_fdsn_client`
             Download and create in one step



   .. py:method:: from_usgs_geomag(request_df: pandas.DataFrame | str | pathlib.Path, **kwargs)
      :classmethod:


      Create MTH5 from USGS geomagnetic observatory data.

      Downloads geomagnetic observatory data from USGS webservices into an
      MTH5 file using a request DataFrame or CSV file.

      :param request_df: Request definition as DataFrame or path to CSV file. Required columns:

                         * **observatory** : str - Observatory code (e.g., 'BOU', 'FRN')
                         * **type** : str - Data type: 'variation', 'adjusted',
                           'quasi-definitive', or 'definitive'
                         * **elements** : str - Geomagnetic elements to retrieve:
                           D, DIST, DST, E, E-E, E-N, F, G, H, SQ, SV, UK1, UK2, UK3, UK4,
                           X, Y, Z
                         * **sampling_period** : int - Sample period in seconds: 1, 60, or 3600
                         * **start** : str - Start time in YYYY-MM-DDThh:mm:ss format (UTC)
                         * **end** : str - End time in YYYY-MM-DDThh:mm:ss format (UTC)
      :type request_df: pd.DataFrame, str, or Path
      :param \*\*kwargs: Additional keyword arguments. HDF5 parameters should be prefixed
                         with 'h5_' (e.g., h5_compression='gzip', h5_compression_opts=1).
      :type \*\*kwargs: dict

      :returns: If interact=False (default), returns Path to created MTH5 file.
                If interact=True, returns MTH5 object with file open.
      :rtype: Path or MTH5

      .. rubric:: Notes

      See USGS Geomagnetism Data web service for more information:
      https://www.usgs.gov/tools/web-service-geomagnetism-data

      .. rubric:: Examples

      Create MTH5 from USGS Boulder observatory using DataFrame:

      >>> import pandas as pd
      >>> from mth5.clients import MakeMTH5
      >>>
      >>> request = pd.DataFrame([{
      ...     'observatory': 'BOU',
      ...     'type': 'variation',
      ...     'elements': 'XYZF',
      ...     'sampling_period': 1,
      ...     'start': '2020-01-01T00:00:00',
      ...     'end': '2020-01-02T00:00:00'
      ... }])
      >>>
      >>> mth5_path = MakeMTH5.from_usgs_geomag(
      ...     request,
      ...     h5_compression='gzip',
      ...     h5_compression_opts=1
      ... )

      Using CSV file:

      >>> mth5_path = MakeMTH5.from_usgs_geomag('requests.csv')

      Multiple observatories and periods:

      >>> request = pd.DataFrame([
      ...     {'observatory': 'BOU', 'type': 'variation',
      ...      'elements': 'XYZF', 'sampling_period': 1,
      ...      'start': '2020-01-01T00:00:00', 'end': '2020-01-02T00:00:00'},
      ...     {'observatory': 'FRN', 'type': 'variation',
      ...      'elements': 'XYZF', 'sampling_period': 60,
      ...      'start': '2020-01-01T00:00:00', 'end': '2020-01-02T00:00:00'}
      ... ])
      >>> mth5_path = MakeMTH5.from_usgs_geomag(request)

      .. seealso::

         :obj:`mth5.io.usgs_geomag.USGSGeomag`
             USGS geomagnetic data client



   .. py:method:: from_zen(data_path: str | pathlib.Path, sample_rates: list[int] = [4096, 1024, 256], calibration_path: str | pathlib.Path | None = None, survey_id: str | None = None, combine: bool = True, **kwargs)
      :classmethod:


      Create MTH5 from Zonge ZEN data files.

      Processes ZEN data files from a directory structure and creates an
      MTH5 file with organized time series data.

      :param data_path: Directory where ZEN data files are stored
      :type data_path: str or Path
      :param sample_rates: Sample rates to include in Hz
      :type sample_rates: list of int, default [4096, 1024, 256]
      :param calibration_path: Path to calibration file (amtant.cal). If None, looks for
                               calibration file in data_path.
      :type calibration_path: str or Path, optional
      :param survey_id: Survey ID to apply to all stations found under data_path. If None,
                        attempts to extract from directory structure.
      :type survey_id: str, optional
      :param combine: If True, combine multiple runs into single run sampled at 1s
      :type combine: bool, default True
      :param \*\*kwargs: Additional keyword arguments. HDF5 parameters should be prefixed
                         with 'h5_' (e.g., h5_compression='gzip', h5_compression_opts=1).
                         Use save_path to specify output directory.
      :type \*\*kwargs: dict

      :returns: Path to created MTH5 file
      :rtype: Path

      .. rubric:: Notes

      ZEN data is typically organized with multiple .Z3D files per station.
      The reader processes these files and organizes them into runs based on
      sampling rate and timing.

      When combine=True, all runs are merged into a single continuous run
      sampled at 1 second intervals, which is useful for long-term datasets.

      .. rubric:: Examples

      Create MTH5 from ZEN data directory:

      >>> from mth5.clients import MakeMTH5
      >>> from pathlib import Path
      >>>
      >>> data_dir = Path("data/zen_survey")
      >>> mth5_path = MakeMTH5.from_zen(
      ...     data_dir,
      ...     sample_rates=[4096, 256],
      ...     survey_id="MT001",
      ...     save_path="output"
      ... )

      With calibration file and HDF5 compression:

      >>> mth5_path = MakeMTH5.from_zen(
      ...     "data/zen_survey",
      ...     calibration_path="data/amtant.cal",
      ...     survey_id="MT001",
      ...     combine=False,
      ...     h5_compression="gzip",
      ...     h5_compression_opts=4
      ... )

      Process all sample rates without combining:

      >>> mth5_path = MakeMTH5.from_zen(
      ...     "data/zen_survey",
      ...     sample_rates=[4096, 1024, 256, 64, 4],
      ...     combine=False
      ... )

      .. seealso::

         :obj:`mth5.io.zen.ZenCollection`
             ZEN data reader



   .. py:method:: from_phoenix(data_path: str | pathlib.Path, mth5_filename: str | None = None, save_path: str | pathlib.Path | None = None, sample_rates: list[int] = [150, 24000], receiver_calibration_dict: str | pathlib.Path | dict | None = None, sensor_calibration_dict: str | pathlib.Path | dict | None = None, **kwargs)
      :classmethod:


      Create MTH5 from Phoenix MTU-5C data files.

      Builds an MTH5 file from Phoenix MTU-5C data with calibration support.
      Requires receiver and sensor calibration files exported from EMPower
      software.

      :param data_path: Directory where Phoenix data files are stored. Can be single station
                        or multiple stations.
      :type data_path: str or Path
      :param mth5_filename: Filename for the MTH5 file. If None, defaults to 'from_phoenix.h5'
      :type mth5_filename: str, optional
      :param save_path: Directory to save MTH5 file. If None, saves to data_path.
      :type save_path: str or Path, optional
      :param sample_rates: Sample rates to include in Hz
      :type sample_rates: list of int, default [150, 24000]
      :param receiver_calibration_dict: Receiver calibration specification:

                                        * str/Path: Directory containing rxcal.json files
                                        * dict: Keys are receiver IDs, values are paths to rxcal.json files
      :type receiver_calibration_dict: str, Path, or dict, optional
      :param sensor_calibration_dict: Sensor calibration specification:

                                      * str/Path: Directory containing scal.json files
                                      * dict: Keys are sensor IDs, values are PhoenixCalibration objects
                                        or paths to scal.json files
      :type sensor_calibration_dict: str, Path, or dict, optional
      :param \*\*kwargs: Additional keyword arguments. HDF5 parameters should be prefixed
                         with 'h5_' (e.g., h5_compression='gzip').
      :type \*\*kwargs: dict

      :returns: Path to created MTH5 file
      :rtype: Path

      .. rubric:: Notes

      Phoenix data requires calibration files exported from EMPower software:

      1. Export rxcal files (receiver calibration) to JSON
      2. Export scal files (sensor calibration) to JSON
      3. Place files in accessible directory
      4. Provide directory path or dict mapping to from_phoenix()

      The method automatically matches calibration files with data based on
      receiver and sensor IDs.

      .. rubric:: Examples

      Basic usage with calibration directories:

      >>> from mth5.clients import MakeMTH5
      >>> from pathlib import Path
      >>>
      >>> data_dir = Path("data/phoenix_survey")
      >>> cal_dir = Path("calibrations")
      >>>
      >>> mth5_path = MakeMTH5.from_phoenix(
      ...     data_dir,
      ...     receiver_calibration_dict=cal_dir / "receivers",
      ...     sensor_calibration_dict=cal_dir / "sensors",
      ...     save_path="output"
      ... )

      With explicit filename and HDF5 compression:

      >>> mth5_path = MakeMTH5.from_phoenix(
      ...     "data/phoenix_survey",
      ...     mth5_filename="MT_survey_2020.h5",
      ...     sample_rates=[150, 24000],
      ...     receiver_calibration_dict="calibrations/receivers",
      ...     sensor_calibration_dict="calibrations/sensors",
      ...     save_path="output",
      ...     h5_compression="gzip",
      ...     h5_compression_opts=4
      ... )

      Using explicit calibration dictionaries:

      >>> receiver_cal = {
      ...     'RX001': Path('cal/rx001_cal.json'),
      ...     'RX002': Path('cal/rx002_cal.json')
      ... }
      >>> sensor_cal = {
      ...     'SN123': phoenix_cal_obj_1,
      ...     'SN124': phoenix_cal_obj_2
      ... }
      >>> mth5_path = MakeMTH5.from_phoenix(
      ...     "data/phoenix_survey",
      ...     receiver_calibration_dict=receiver_cal,
      ...     sensor_calibration_dict=sensor_cal
      ... )

      .. seealso::

         :obj:`mth5.io.phoenix.PhoenixClient`
             Phoenix data reader

         :obj:`mth5.io.phoenix.PhoenixCalibration`
             Calibration file handler



   .. py:method:: from_lemi424(data_path: str | pathlib.Path, survey_id: str, station_id: str, mth5_filename: str = 'from_lemi424.h5', save_path: str | pathlib.Path = Path().cwd(), **kwargs)
      :classmethod:


      Create MTH5 from LEMI-424 long period data.

      Builds an MTH5 file from LEMI-424 instrument data on a station-by-station
      basis. LEMI data has limited metadata, so survey and station IDs must
      be provided.

      :param data_path: Directory where LEMI-424 data files are stored. Can be single
                        station or full directory.
      :type data_path: str or Path
      :param survey_id: Survey ID to apply to all stations
      :type survey_id: str
      :param station_id: Station ID for this station's data
      :type station_id: str
      :param mth5_filename: Filename for the MTH5 output file
      :type mth5_filename: str, default 'from_lemi424.h5'
      :param save_path: Directory to save MTH5 file
      :type save_path: str or Path, default current directory
      :param \*\*kwargs: Additional keyword arguments. HDF5 parameters should be prefixed
                         with 'h5_' (e.g., h5_compression='gzip').
      :type \*\*kwargs: dict

      :returns: Path to created MTH5 file
      :rtype: Path

      .. rubric:: Notes

      LEMI-424 is a long-period magnetotelluric instrument. Data files have
      limited embedded metadata, requiring manual specification of survey
      and station information.

      Process each station individually due to minimal automatic metadata
      extraction capabilities.

      .. rubric:: Examples

      Create MTH5 from LEMI-424 data:

      >>> from mth5.clients import MakeMTH5
      >>> from pathlib import Path
      >>>
      >>> data_dir = Path("data/lemi_mt01")
      >>> mth5_path = MakeMTH5.from_lemi424(
      ...     data_dir,
      ...     survey_id='MT2020',
      ...     station_id='MT01',
      ...     save_path="output"
      ... )

      With HDF5 compression:

      >>> mth5_path = MakeMTH5.from_lemi424(
      ...     "data/lemi_mt01",
      ...     survey_id='MT2020',
      ...     station_id='MT01',
      ...     mth5_filename='MT2020_MT01.h5',
      ...     h5_compression='gzip',
      ...     h5_compression_opts=1
      ... )

      Multiple stations (process individually):

      >>> for station in ['MT01', 'MT02', 'MT03']:
      ...     data_dir = Path(f"data/lemi_{station.lower()}")
      ...     mth5_path = MakeMTH5.from_lemi424(
      ...         data_dir,
      ...         survey_id='MT2020',
      ...         station_id=station,
      ...         mth5_filename=f'MT2020_{station}.h5',
      ...         save_path="output"
      ...     )

      .. seealso::

         :obj:`mth5.io.lemi424.LEMI424Client`
             LEMI-424 data reader



   .. py:method:: from_metronix(data_path: str | pathlib.Path, sample_rates: list[float] = [128], mth5_filename: str | None = None, save_path: str | pathlib.Path | None = None, run_name_zeros: int = 0, **kwargs)
      :classmethod:


      Create MTH5 from Metronix Geophysics ATSS + JSON files.

      Builds an MTH5 file from Metronix data in their new folder structure
      format with ATSS time series and JSON metadata files.

      :param data_path: Highest level directory to archive data from, usually the survey
                        level. For single station, use station folder path.
      :type data_path: str or Path
      :param sample_rates: Sample rates to archive in samples/second
      :type sample_rates: list of float, default [128]
      :param mth5_filename: Filename for the MTH5 file. If None, automatically generated from
                            survey/station information.
      :type mth5_filename: str, optional
      :param save_path: Directory to save MTH5 file. If None, saves to current working
                        directory.
      :type save_path: str or Path, optional
      :param run_name_zeros: Number of zeros for zero-padding in run names. Run names formatted
                             as 'sr{sample_rate}_{run_id:0{run_name_zeros}}'. If 0, uses
                             original run names (e.g., 'run_0001').
      :type run_name_zeros: int, default 0
      :param \*\*kwargs: Additional keyword arguments. HDF5 parameters should be prefixed
                         with 'h5_' (e.g., h5_compression='gzip').
      :type \*\*kwargs: dict

      :returns: Path to created MTH5 file
      :rtype: Path

      .. rubric:: Notes

      Metronix Geophysics uses a specific folder structure with ATSS binary
      files and JSON metadata. The reader processes this structure and
      organizes data by survey, station, and run.

      Run naming can be customized with run_name_zeros:
      - run_name_zeros=0: Keep original names like 'run_0001'
      - run_name_zeros=4: Format as 'sr128_0001'
      - run_name_zeros=2: Format as 'sr128_01'

      .. rubric:: Examples

      Create MTH5 from Metronix survey data:

      >>> from mth5.clients import MakeMTH5
      >>> from pathlib import Path
      >>>
      >>> data_dir = Path("data/metronix_survey")
      >>> mth5_path = MakeMTH5.from_metronix(
      ...     data_dir,
      ...     sample_rates=[128, 4096],
      ...     save_path="output"
      ... )

      With custom run naming and compression:

      >>> mth5_path = MakeMTH5.from_metronix(
      ...     "data/metronix_survey",
      ...     sample_rates=[128],
      ...     mth5_filename="survey_2020.h5",
      ...     run_name_zeros=4,
      ...     h5_compression="gzip",
      ...     h5_compression_opts=4
      ... )

      Single station with original run names:

      >>> mth5_path = MakeMTH5.from_metronix(
      ...     "data/metronix_survey/Station_001",
      ...     sample_rates=[128, 512],
      ...     run_name_zeros=0,
      ...     save_path="output"
      ... )

      Multiple sample rates with formatted names:

      >>> mth5_path = MakeMTH5.from_metronix(
      ...     "data/metronix_survey",
      ...     sample_rates=[32, 128, 512, 4096],
      ...     run_name_zeros=3,
      ...     mth5_filename="mt_data.h5"
      ... )

      .. seealso::

         :obj:`mth5.io.metronix.MetronixClient`
             Metronix data reader



   .. py:method:: from_nims(data_path, sample_rates=[4096, 1024, 256], save_path=None, calibration_path=None, survey_id=None, combine=True, **kwargs)
      :classmethod:


      Create an MTH5 from nims data.

      Any H5 file parameters like compression, shuffle, etc need to have a
      prefix of 'h5'. For example h5_compression='gzip'.

      >>> MakeMTH5.from_nims(
          data_path, **{'h5_compression_opts': 1}
          )

      :param data_path: directory to where data are stored
      :type data_path: Path, str
      :param sample_rates: sample rates to include,
       defaults to [4096, 1024, 256]
      :type sample_rates: list, optional
      :param save_path: path to save H5 file to, defaults to None which will
       place the file in `data_path`
      :type save_path: str or Path, optional
      :param calibration_path: path to calibration file amtant.cal,
       defaults to None
      :type calibration_path: str or Path, optional
      :param survey_id: survey ID to apply to all station found under
       `data_path`, defaults to None
      :type survey_id: string
      :param combine: if True combine the runs into a single run sampled at 1s,
       defaults to True
      :type combine: bool
      :return: MTH5 file name
      :rtype: Path




