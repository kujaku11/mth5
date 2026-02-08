mth5.data.station_config
========================

.. py:module:: mth5.data.station_config

.. autoapi-nested-parse::

   This module contains tools for building MTH5 files from synthetic data.

   Development Notes:
     - These tools are a work in progress and ideally will be able to yield
     generalize to more than just the legacy EMTF ascii datasets that they
     initially served.

   Definitions used in the creation of synthetic mth5 files.

   Survey level: 'mth5_path', Path to output h5

   Station level: mt_metadata Station() object with station info.
     - the `id` field (name of the station) is required.
     - other station metadata can be added
     - channel_nomenclature
       - The channel_nomenclature was previously stored at the run level.  It makes more sense to store
        this info at the station level, as the only reason the nomenclature would change (that I can
        think of) would be if the acquistion system changed, in which case it would make the most
        sense to initialize a new station object.

   Run level: 'columns', :channel names as a list; ["hx", "hy", "hz", "ex", "ey"]
   Run level: 'raw_data_path', Path to ascii data source
   Run level: 'noise_scalars', dict keyed by channel, default is zero,
   Run level: 'nan_indices', iterable of integers, where to put nan [
   Run level: 'filters', dict of filters keyed by columns
   Run level: 'run_id', name of the run
   Run level: 'sample_rate', 1.0




Attributes
----------

.. autoapisummary::

   mth5.data.station_config.ASCII_DATA_PATH
   mth5.data.station_config.FILTERS


Classes
-------

.. autoapisummary::

   mth5.data.station_config.SyntheticRun
   mth5.data.station_config.SyntheticStation
   mth5.data.station_config.LegacyEMTFAsciiFile


Functions
---------

.. autoapisummary::

   mth5.data.station_config.make_filters
   mth5.data.station_config.make_station_01
   mth5.data.station_config.make_station_02
   mth5.data.station_config.make_station_03
   mth5.data.station_config.make_station_04
   mth5.data.station_config.main


Module Contents
---------------

.. py:data:: ASCII_DATA_PATH

.. py:function:: make_filters(as_list: Optional[bool] = False) -> Union[dict, list]

       Creates a collection of filters
   Because the synthetic data from EMTF are already in mV/km and nT, no calibration filters are required.
    The filters here are placeholders to show where instrument response function information would get assigned.

   :param as_list: If True we return a list, False return a dict
   :type as_list: bool
   :return filters_list: Filters for populating the filters lists of synthetic data
   :rtype filters_list: Union[List, Dict]


.. py:data:: FILTERS

.. py:class:: SyntheticRun(id: str, sample_rate: float, channels: List[str], raw_data_path: Optional[Union[str, pathlib.Path]] = None, noise_scalars: Optional[dict] = None, nan_indices: Optional[dict] = None, filters: Optional[dict] = None, start: Optional[str] = None, timeseries_dataframe: Optional[pandas.DataFrame] = None, data_source: str = 'legacy emtf ascii')

   Bases: :py:obj:`object`


   Place to store information that will be needed to initialize and MTH5 Run object.

   Initially this class worked only with the synthetic ASCII data from legacy EMTF.


   .. py:attribute:: channels


   .. py:attribute:: noise_scalars
      :value: None



   .. py:attribute:: run_metadata


.. py:class:: SyntheticStation(station_metadata: mt_metadata.timeseries.Station, mth5_name: Optional[Union[str, pathlib.Path]] = None, channel_nomenclature_keyword: mt_metadata.processing.aurora.channel_nomenclature.SupportedNomenclatureEnum = SupportedNomenclatureEnum.default)

   Bases: :py:obj:`object`


   Class used to contain information needed to generate MTH5 file from synthetic data.

   TODO: could add channel_nomenclature to this obj (instead of run) but would need to decide that
    runs cannot change channel nomenclature first. If that were decided, the channel_map() could go here as well.



   .. py:attribute:: station_metadata


   .. py:attribute:: runs
      :value: []



   .. py:attribute:: mth5_name
      :value: None



   .. py:attribute:: channel_nomenclature_keyword


   .. py:property:: channel_nomenclature


.. py:function:: make_station_01(channel_nomenclature: mt_metadata.processing.aurora.channel_nomenclature.SupportedNomenclatureEnum = SupportedNomenclatureEnum.default) -> SyntheticStation

       This method prepares the metadata needed to generate an mth5 with syntheric data.

   :param channel_nomenclature: Must be one of the nomenclatures defined in SupportedNomenclatureEnum
   :type channel_nomenclature: str

   :return: Object with all info needed to generate MTH5 file from synthetic data.
   :rtype: SyntheticStation



.. py:function:: make_station_02(channel_nomenclature: mt_metadata.processing.aurora.channel_nomenclature.SupportedNomenclatureEnum = SupportedNomenclatureEnum.default) -> SyntheticStation

   Just like station 1, but the data are different

   :param channel_nomenclature: Must be one of the nomenclatures defined in SupportedNomenclatureEnum
   :type channel_nomenclature: SupportedNomenclatureEnum
   :return: Object with all info needed to generate MTH5 file from synthetic data.
   :rtype: SyntheticStation



.. py:function:: make_station_03(channel_nomenclature: mt_metadata.processing.aurora.channel_nomenclature.SupportedNomenclatureEnum = SupportedNomenclatureEnum.default) -> SyntheticStation

   Create a synthetic station with multiple runs.  Rather than generate fresh
   synthetic data, we just reuse test1.asc for each run.

   :param channel_nomenclature: Literal, Must be one of the nomenclatures defined in "channel_nomenclatures.json"
   :type channel_nomenclature: SupportedNomenclatureEnum
   :rtype: SyntheticStation
   :return: Object with all info needed to generate MTH5 file from synthetic data.



.. py:function:: make_station_04(channel_nomenclature: mt_metadata.processing.aurora.channel_nomenclature.SupportedNomenclatureEnum = SupportedNomenclatureEnum.default) -> SyntheticStation

   Just like station 01, but data are resampled to 8Hz

   :param channel_nomenclature: Literal, Must be one of the nomenclatures defined in "channel_nomenclatures.json"
   :type channel_nomenclature: SupportedNomenclatureEnum
   :rtype: SyntheticStation
   :return: Object with all info needed to generate MTH5 file from synthetic data.


.. py:class:: LegacyEMTFAsciiFile(file_path: pathlib.Path)

   This class can be used to interact with the legacy synthetic data files
   that were originally in EMTF.

   Development Notes:
    As of 2025-02-03 the only LegacyEMTFAsciiFile date sources are sampled at 1Hz.
    One-off upsampling can be handled in this class if the requested sample rate differs.



   .. py:attribute:: IMPLICIT_SAMPLE_RATE
      :value: 1.0



   .. py:attribute:: file_path


   .. py:method:: load_dataframe(channel_names: list, sample_rate: float) -> pandas.DataFrame

      Loads an EMTF legacy ASCII time series into a dataframe.

      These files have an awkward whitespace separator, and also need to have the
       electric field channels inverted to fix a phase swap.

      :param channel_names: The names of the channels in the legacy EMTF file, in order.
      :type channel_names: list
      :param sample_rate: The sample rate of the output time series in Hz.
      :type sample_rate: float

      :return df: The labelled time series from the legacy EMTF file.
      :rtype df: pd.DataFrame




.. py:function:: main()

