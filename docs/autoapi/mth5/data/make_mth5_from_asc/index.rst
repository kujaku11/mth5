mth5.data.make_mth5_from_asc
============================

.. py:module:: mth5.data.make_mth5_from_asc

.. autoapi-nested-parse::

   Created on Fri Jun 25 16:03:21 2021

   @author: jpeacock

   This module is concerned with creating mth5 files from the synthetic test data
    that originally came from EMTF -- test1.asc and test2.asc.  Each ascii file
    represents five channels of data sampled at 1Hz at a synthetic station.

   TODO: Separate the handling of legacy EMTF data files, such as
    reading into a dataframe from oddly delimited data, as well as flipping polarities of
    the electric channels (possibly due to a baked in sign convention error in the legacy
    data), so that a simple dataframe can be passed.  That will make the methods here more
     easily generalize to work with other dataframes.  That would be useful in future when
     we creating synthetic data at arbitrary sample rate.

   Development Notes:
    Mirroring the original ascii files are:
    data/test1.h5
    data/test2.h5
    data/test12rr.h5

    Also created are some files with the same data but other channel_nomenclature schemes:
    data/test12rr_LEMI34.h5
    data/test1_LEMI12.h5

    - 20231103: Added an 8Hz up-sampled version of test1.  No spectral content was added
    so the band between the old and new Nyquist frequencies is bogus.





Attributes
----------

.. autoapisummary::

   mth5.data.make_mth5_from_asc.synthetic_test_paths
   mth5.data.make_mth5_from_asc.MTH5_PATH


Functions
---------

.. autoapisummary::

   mth5.data.make_mth5_from_asc.create_run_ts_from_synthetic_run
   mth5.data.make_mth5_from_asc.get_time_series_dataframe
   mth5.data.make_mth5_from_asc.create_mth5_synthetic_file
   mth5.data.make_mth5_from_asc.create_test1_h5
   mth5.data.make_mth5_from_asc.create_test2_h5
   mth5.data.make_mth5_from_asc.create_test1_h5_with_nan
   mth5.data.make_mth5_from_asc.create_test12rr_h5
   mth5.data.make_mth5_from_asc.create_test3_h5
   mth5.data.make_mth5_from_asc.create_test4_h5
   mth5.data.make_mth5_from_asc.main


Module Contents
---------------

.. py:data:: synthetic_test_paths

.. py:data:: MTH5_PATH

.. py:function:: create_run_ts_from_synthetic_run(run: mth5.data.station_config.SyntheticRun, df: pandas.DataFrame, channel_nomenclature: mt_metadata.processing.aurora.ChannelNomenclature) -> mth5.timeseries.RunTS

   Loop over channels of synthetic data in df and make ChannelTS objects.

   :param run: One-off data structure with information mth5 needs to initialize. Specifically sample_rate, filters.
   :type run: mth5.data.station_config.SyntheticRun
   :param df: time series data in columns labelled from ["ex", "ey", "hx", "hy", "hz"]
   :type df: pandas.DataFrame
   :param channel_nomenclature : Keyword corresponding to channel nomenclature mapping
   in CHANNEL_MAPS variable from channel_nomenclature.py module in mt_metadata.
   Supported values include ['default', 'lemi12', 'lemi34', 'phoenix123']
   :type channel_nomenclature : string

   :return runts: MTH5 run time series object, data and metadata bound into one.
   :rtype runts: RunTS



.. py:function:: get_time_series_dataframe(run: mth5.data.station_config.SyntheticRun, source_folder: Union[pathlib.Path, str], add_nan_values: Optional[bool] = False) -> pandas.DataFrame

   Returns time series data in a dataframe with columns named for EM field component.

   Up-samples data to run.sample_rate, which is treated as in integer.
   Only tested for 8, to make 8Hz data for testing.  If run.sample_rate is default (1.0)
   then no up-sampling takes place.

   TODO: Move noise, and nan addition out of this method.

   :type run: mth5.data.station_config.SyntheticRun
   :param run: Information needed to define/create the run
   :type source_folder: Optional[Union[pathlib.Path, str]]
   :param source_folder: Where to load the ascii time series from.  This overwrites any
   previous value that may have been stored in the SyntheticRun
   :type add_nan_values: bool
   :param add_nan_values: If True, add some NaN, if False, do not add Nan.
   :rtype df: pandas.DataFrame
   :return df: The time series data for the synthetic run



.. py:function:: create_mth5_synthetic_file(station_cfgs: List[mth5.data.station_config.SyntheticStation], mth5_name: Union[pathlib.Path, str], target_folder: Optional[Union[pathlib.Path, str]] = '', source_folder: Union[pathlib.Path, str] = '', plot: bool = False, add_nan_values: bool = False, file_version: Literal['0.1.0', '0.2.0'] = '0.1.0', force_make_mth5: bool = True, survey_metadata: Optional[mt_metadata.timeseries.Survey] = None)

   Creates an MTH5 from synthetic data.

   Development Notes:
    20250203: This function could be made more general, so that it operates on dataframes and legacy emtf ascii files.

   :param station_cfgs: Iterable of objects of type SyntheticStation. These are one-off
   data structure used to hold information mth5 needs to initialize, specifically
   sample_rate, filters, etc.
   :type station_cfgs: List[SyntheticStation]
   :param mth5_name: Where the mth5 will be stored.  This is generated by the station_config,
   but may change in this method based on add_nan_values or channel_nomenclature
   :type mth5_name: Union[pathlib.Path, str]
   :param target_folder: Where the mth5 file will be stored
   :type target_folder: Optional[Union[pathlib.Path, str]]
   :param source_folder:  Where the ascii source data are stored
   :type source_folder: Optional[Union[pathlib.Path, str]] = "",
   :param plot: Set to false unless you want to look at a plot of the time series
   :type plot: bool
   :param add_nan_values: If true, some np.nan are sprinkled into the time series.  Intended to be used for tests.
   :type add_nan_values: bool
   :param file_version: One of the supported mth5 file versions.  This is the version of mth5 to create.
   :type file_version: Literal["0.1.0", "0.2.0"] = "0.1.0",
   :param force_make_mth5: If set to true, the file will be made, even if it already exists.
   If false, and file already exists, skip the make job.
   :type force_make_mth5: bool
   :param survey_metadata: Option to provide survey metadata, otherwise it will be created.
   :type survey_metadata: Survey
   :return: The path to the stored h5 file.
   :rtype: mth5_path: pathlib.Path



.. py:function:: create_test1_h5(file_version: Optional[str] = '0.1.0', channel_nomenclature: Optional[str] = 'default', target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH, source_folder: Optional[Union[str, pathlib.Path]] = '', force_make_mth5: Optional[bool] = True) -> pathlib.Path

   Creates an MTH5 file for a single station named "test1".

   :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
   :type file_version: str
   :type channel_nomenclature: Optional[str]
   :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
   from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
   A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
   :type target_folder: Optional[Union[str, pathlib.Path]]
   :param target_folder: Where the mth5 file will be stored
   :type source_folder: Optional[Union[str, pathlib.Path]]
   :param source_folder:  Where the ascii source data are stored
   :type force_make_mth5: bool
   :param force_make_mth5: If set to true, the file will be made, even if it already exists.
   If false, and file already exists, skip the make job.
   :rtype: pathlib.Path
   :return: the path to the mth5 file



.. py:function:: create_test2_h5(file_version: Optional[str] = '0.1.0', channel_nomenclature: Optional[str] = 'default', target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH, source_folder: Optional[Union[str, pathlib.Path]] = '', force_make_mth5: Optional[bool] = True) -> pathlib.Path

   Creates an MTH5 file for a single station named "test2".

   :type file_version: str
   :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
   :type channel_nomenclature: Optional[str]
   :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
   from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
   A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
   :type target_folder: Optional[str, pathlib.Path]
   :param target_folder: Where the mth5 file will be stored
   :type source_folder: Optional[str, pathlib.Path]
   :param source_folder:  Where the ascii source data are stored
   :type force_make_mth5: bool
   :param force_make_mth5: If set to true, the file will be made, even if it already exists.
   If false, and file already exists, skip the make job.
   :rtype: pathlib.Path
   :return: the path to the mth5 file


.. py:function:: create_test1_h5_with_nan(file_version: Optional[str] = '0.1.0', channel_nomenclature: Optional[str] = 'default', target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH, source_folder: Optional[Union[str, pathlib.Path]] = '', force_make_mth5: Optional[bool] = True) -> pathlib.Path

   Creates an MTH5 file for a single station named "test1" with some nan values.

   :type file_version: str
   :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
   :type channel_nomenclature: Optional[str]
   :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
   from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
   A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
   :type target_folder: Optional[str, pathlib.Path]
   :param target_folder: Where the mth5 file will be stored
   :type source_folder: Optional[str, pathlib.Path]
   :param source_folder:  Where the ascii source data are stored
   :rtype: pathlib.Path
   :return: the path to the mth5 file


.. py:function:: create_test12rr_h5(file_version: Optional[str] = '0.1.0', channel_nomenclature: Optional[str] = 'default', target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH, source_folder: Optional[Union[str, pathlib.Path]] = '', force_make_mth5: Optional[bool] = True) -> pathlib.Path

   Creates an MTH5 file with data from two stations station named "test1" and "test2".

   :type file_version: str
   :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
   :type channel_nomenclature: Optional[str]
   :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
   from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
   A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
   :type target_folder: Optional[str, pathlib.Path]
   :param target_folder: Where the mth5 file will be stored
   :type source_folder: Optional[str, pathlib.Path]
   :param source_folder:  Where the ascii source data are stored
   :rtype: pathlib.Path
   :return: the path to the mth5 file


.. py:function:: create_test3_h5(file_version: Optional[str] = '0.1.0', channel_nomenclature: Optional[str] = 'default', target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH, source_folder: Optional[Union[str, pathlib.Path]] = '', force_make_mth5: Optional[bool] = True) -> pathlib.Path

   Creates an MTH5 file for a single station named "test3".
   This example has several runs and can be used to test looping over runs.

   :type file_version: str
   :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
   :type channel_nomenclature: Optional[str]
   :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
   from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
   A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
   :type target_folder: Optional[str, pathlib.Path]
   :param target_folder: Where the mth5 file will be stored
   :type source_folder: Optional[str, pathlib.Path]
   :param source_folder:  Where the ascii source data are stored
   :type force_make_mth5: bool
   :param force_make_mth5: If set to true, the file will be made, even if it already exists.
   If false, and file already exists, skip the make job.
   :rtype: pathlib.Path
   :return: the path to the mth5 file


.. py:function:: create_test4_h5(file_version: Optional[str] = '0.1.0', channel_nomenclature: Optional[str] = 'default', target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH, source_folder: Optional[Union[str, pathlib.Path]] = '', force_make_mth5: Optional[bool] = True) -> pathlib.Path

   Creates an MTH5 file for a single station named "test1", data are up-sampled to 8Hz from
   original 1 Hz.

   Note: Because the 8Hz data are derived from the 1Hz, only frequencies below 0.5Hz
   will have valid TFs that yield the apparent resistivity of the synthetic data (100 Ohm-m).

   :type file_version: str
   :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
   :type channel_nomenclature: Optional[str]
   :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
   from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
   A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
   :type target_folder: Optional[str, pathlib.Path]
   :param target_folder: Where the mth5 file will be stored
   :type source_folder: Optional[str, pathlib.Path]
   :param source_folder:  Where the ascii source data are stored
   :rtype: pathlib.Path
   :return: the path to the mth5 file


.. py:function:: main(file_version='0.1.0')

   Allow the module to be called from the command line


