mth5.utils.helpers
==================

.. py:module:: mth5.utils.helpers


Attributes
----------

.. autoapisummary::

   mth5.utils.helpers.T


Functions
---------

.. autoapisummary::

   mth5.utils.helpers.path_or_mth5_object
   mth5.utils.helpers.get_version
   mth5.utils.helpers.get_channel_summary
   mth5.utils.helpers.add_filters
   mth5.utils.helpers.initialize_mth5
   mth5.utils.helpers.read_back_data
   mth5.utils.helpers.get_compare_dict
   mth5.utils.helpers.station_in_mth5
   mth5.utils.helpers.survey_in_mth5


Module Contents
---------------

.. py:data:: T

.. py:function:: path_or_mth5_object(func: Callable[Ellipsis, T]) -> Callable[Ellipsis, T]

   Decorator allowing functions to accept MTH5 file paths or MTH5 objects.

   Transparently converts file paths to MTH5 objects, opens the file,
   and passes the MTH5 object to the decorated function.

   :param func: A function that takes an MTH5 object as its first argument.
                Signature: func(mth5_obj: MTH5, *args, **kwargs) -> T
   :type func: Callable

   :returns: Wrapped function accepting str/Path or MTH5 as first argument.
   :rtype: Callable

   :raises TypeError: If first argument is not a string, pathlib.Path, or MTH5 object.

   .. rubric:: Notes

   The decorated function can be called with either:
   - A file path string or pathlib.Path
   - An MTH5 object

   When given a file path, the decorator automatically opens the file
   in 'append' mode by default, unless overridden in kwargs.

   TODO: add support for file_version in kwargs

   .. rubric:: Examples

   Decorate a function to work with both paths and objects::

       @path_or_mth5_object
       def get_metadata(m: MTH5) -> dict:
           return m.survey_group.metadata.to_dict()

       # Call with file path
       metadata = get_metadata('/path/to/file.mth5')

       # Call with MTH5 object
       with MTH5() as m:
           m.open_mth5('/path/to/file.mth5', mode='r')
           metadata = get_metadata(m)


.. py:function:: get_version(m: str | pathlib.Path | mth5.mth5.MTH5) -> str

   Get the file version from an MTH5 file.

   :param m: Path to MTH5 file or MTH5 object.
   :type m: str | pathlib.Path | MTH5

   :returns: File version string (e.g., '0.1.0', '0.2.0').
   :rtype: str

   .. rubric:: Examples

   Get version from file path::

       >>> version = get_version('/path/to/file.mth5')
       >>> print(version)
       '0.2.0'

   Get version from MTH5 object::

       >>> with MTH5() as m:
       ...     m.open_mth5('/path/to/file.mth5')
       ...     version = get_version(m)


.. py:function:: get_channel_summary(m: str | pathlib.Path | mth5.mth5.MTH5, show: bool = True) -> Any

   Get channel summary from MTH5 file as pandas DataFrame.

   Retrieves the channel summary table and converts to DataFrame.
   Automatically re-summarizes if the summary appears incomplete.

   :param m: Path to MTH5 file or MTH5 object.
   :type m: str | pathlib.Path | MTH5
   :param show: Whether to log the summary DataFrame to console.
   :type show: bool, default True

   :returns: Channel summary with station, run, and channel information.
   :rtype: pandas.DataFrame

   .. warning::

      If the summary appears incomplete, the channel summary table is
      re-summarized which may take time for large files.

   .. rubric:: Examples

   Get channel summary from file path::

       >>> df = get_channel_summary('/path/to/file.mth5')
       >>> print(df.shape)
       (42, 8)

   Get summary without logging::

       >>> df = get_channel_summary('/path/to/file.mth5', show=False)


.. py:function:: add_filters(m: str | pathlib.Path | mth5.mth5.MTH5, filters_list: list[Any], survey_id: str = '') -> None

   Add filter objects to MTH5 file.

   Adds a list of filter objects to the MTH5 file's filter group.
   Automatically selects the appropriate filters group based on file version.

   :param m: Path to MTH5 file or MTH5 object.
   :type m: str | pathlib.Path | MTH5
   :param filters_list: List of filter objects to add. Each filter should have a 'name'
                        attribute and be compatible with the filters group.
   :type filters_list: list
   :param survey_id: Survey ID for file version 0.2.0. Required for version 0.2.0,
                     ignored for version 0.1.0.
   :type survey_id: str, default ''

   :raises AttributeError: If filter objects lack required attributes.
   :raises ValueError: If survey_id is not found in version 0.2.0 files.

   .. rubric:: Notes

   File version 0.1.0 stores filters globally.
   File version 0.2.0 stores filters per survey.

   .. rubric:: Examples

   Add filters to MTH5 file::

       >>> from mth5.timeseries import Filter
       >>> filters = [Filter(name='test_filter')]
       >>> add_filters('/path/to/file.mth5', filters)

   Add survey-specific filters (version 0.2.0)::

       >>> add_filters('/path/to/file.mth5', filters, survey_id='MT01')


.. py:function:: initialize_mth5(h5_path: str | pathlib.Path, mode: str = 'a', file_version: str = '0.1.0') -> mth5.mth5.MTH5

   Initialize and open an MTH5 file for reading or writing.

   Creates or opens an MTH5 file with specified file version.
   Optionally removes existing files before write operations.

   :param h5_path: Path to MTH5 file. Created if it doesn't exist.
   :type h5_path: str | pathlib.Path
   :param mode: File access mode:
                - 'r': read-only
                - 'w': write (overwrites existing file)
                - 'a': append/read-write
   :type mode: {'r', 'w', 'a'}, default 'a'
   :param file_version: MTH5 file format version.
   :type file_version: {'0.1.0', '0.2.0'}, default '0.1.0'

   :returns: Initialized and opened MTH5 object.
   :rtype: MTH5

   .. warning::

      When mode='w' and file exists, all open h5 files are closed before
      removal. This may affect other processes using HDF5 files.

   .. rubric:: Examples

   Create a new MTH5 file::

       >>> m = initialize_mth5('/path/to/file.mth5', mode='w')
       >>> m.file_version
       '0.1.0'
       >>> m.close_mth5()

   Open existing file for appending::

       >>> m = initialize_mth5('/path/to/file.mth5', mode='a')
       >>> m.add_station('MT001')
       >>> m.close_mth5()

   Open file with version 0.2.0 schema::

       >>> m = initialize_mth5('/path/to/file.mth5', file_version='0.2.0')


.. py:function:: read_back_data(mth5_path: str | pathlib.Path, station_id: str, run_id: str, survey: str | None = None, close_mth5: bool = True, return_objects: list[str] | None = None) -> dict[str, Any]

   Read station/run data from MTH5 file for testing and validation.

   Helper function to confirm MTH5 file accessibility and validate
   that data dimensions match expectations.

   :param mth5_path: Full path to MTH5 file to read.
   :type mth5_path: str | pathlib.Path
   :param station_id: Station identifier (e.g., 'PKD', 'MT001').
   :type station_id: str
   :param run_id: Run identifier (e.g., '001', '1').
   :type run_id: str
   :param survey: Survey identifier. Required for file version 0.2.0.
   :type survey: str, optional
   :param close_mth5: Whether to close MTH5 object after reading.
                      Set to False if you need to access the object later.
   :type close_mth5: bool, default True
   :param return_objects: Specifies what objects to return. Options:
                          - 'run': RunGroup object
                          - 'run_ts': RunTS time series object
                          If None, returns empty dict with only mth5_obj if close_mth5=False.
   :type return_objects: list of str, optional

   :returns: Dictionary containing requested objects:
             - 'run': RunGroup (if 'run' in return_objects)
             - 'run_ts': RunTS (if 'run_ts' in return_objects)
             - 'mth5_obj': MTH5 (if close_mth5=False)
   :rtype: dict

   .. warning::

      If close_mth5=False, the MTH5 object must be manually closed
      to avoid resource leaks.

   .. rubric:: Notes

   This is primarily a testing utility. Data shape is logged to console.

   .. rubric:: Examples

   Read run data and close immediately::

       >>> result = read_back_data(
       ...     '/path/to/file.mth5',
       ...     'PKD',
       ...     '001',
       ...     return_objects=['run_ts']
       ... )
       >>> ts = result['run_ts']
       >>> print(ts.dataset.shape)

   Read data and keep MTH5 object open::

       >>> result = read_back_data(
       ...     '/path/to/file.mth5',
       ...     'MT001',
       ...     '1',
       ...     survey='survey_01',
       ...     close_mth5=False,
       ...     return_objects=['run', 'run_ts']
       ... )
       >>> run = result['run']
       >>> m = result['mth5_obj']
       >>> # ... use objects ...
       >>> m.close_mth5()

   TODO: add path_or_mth5_decorator to this function


.. py:function:: get_compare_dict(input_dict: dict[str, Any]) -> dict[str, Any]

   Remove MTH5-specific metadata attributes for comparison.

   Removes internal attributes added by MTH5 that may interfere
   with dictionary comparisons between metadata objects.

   :param input_dict: Dictionary to clean, typically metadata dictionary.
   :type input_dict: dict

   :returns: Dictionary with MTH5 internal attributes removed.
             Original dict is modified in-place.
   :rtype: dict

   .. rubric:: Notes

   Removed attributes:
   - hdf5_reference: HDF5 object reference (internal)
   - mth5_type: MTH5 data type marker (internal)

   .. rubric:: Examples

   Clean metadata dictionary before comparison::

       >>> metadata = {
       ...     'id': 'station_001',
       ...     'latitude': 45.5,
       ...     'hdf5_reference': <h5py reference>,
       ...     'mth5_type': 'Station'
       ... }
       >>> clean = get_compare_dict(metadata)
       >>> print(clean)
       {'id': 'station_001', 'latitude': 45.5}

   Safe to call with incomplete dicts::

       >>> metadata = {'id': 'station_001'}
       >>> clean = get_compare_dict(metadata)  # No error if keys absent


.. py:function:: station_in_mth5(m: str | pathlib.Path | mth5.mth5.MTH5, station_id: str, survey_id: str | None = None) -> bool

   Check if a station exists in MTH5 file.

   Determines whether a station with the given ID is present
   in the MTH5 file using the groups list.

   :param m: Path to MTH5 file or MTH5 object.
   :type m: str | pathlib.Path | MTH5
   :param station_id: Station identifier (e.g., 'PKD', 'MT001').
   :type station_id: str
   :param survey_id: Survey identifier. Required for file version 0.2.0,
                     ignored for version 0.1.0.
   :type survey_id: str, optional

   :returns: True if station exists, False otherwise.
   :rtype: bool

   :raises NotImplementedError: If file version is not 0.1.0 or 0.2.0.

   .. rubric:: Notes

   File version 0.1.0 has global stations group.
   File version 0.2.0 has per-survey stations groups.

   Alternative method: Use channel_summary DataFrame::

       df = m.channel_summary.to_dataframe()
       station_exists = station_id in df['Station'].unique()

   .. rubric:: Examples

   Check if station exists (file version 0.1.0)::

       >>> exists = station_in_mth5('/path/to/file.mth5', 'PKD')
       >>> print(exists)
       True

   Check in version 0.2.0 with survey ID::

       >>> exists = station_in_mth5(
       ...     '/path/to/file.mth5',
       ...     'MT001',
       ...     survey_id='survey_01'
       ... )


.. py:function:: survey_in_mth5(m: str | pathlib.Path | mth5.mth5.MTH5, survey_id: str | None = None) -> bool

   Check if a survey exists in MTH5 file.

   Determines whether a survey with the given ID exists in the MTH5 file.
   Behavior varies by file version: 0.1.0 has a single survey, while
   0.2.0 supports multiple surveys.

   :param m: Path to MTH5 file or MTH5 object.
   :type m: str | pathlib.Path | MTH5
   :param survey_id: Survey identifier. For file version 0.1.0, compared against the
                     global survey ID. For version 0.2.0, checked in surveys group.
   :type survey_id: str, optional

   :returns: True if survey exists, False otherwise.
   :rtype: bool

   :raises NotImplementedError: If file version is not 0.1.0 or 0.2.0.

   .. rubric:: Notes

   File version 0.1.0 has a single survey with fixed ID.
   File version 0.2.0 supports multiple named surveys.

   Alternative method: Use channel_summary DataFrame::

       df = m.channel_summary.to_dataframe()
       surveys = df['Survey'].unique()
       survey_exists = survey_id in surveys

   .. rubric:: Examples

   Check if survey exists (file version 0.1.0)::

       >>> exists = survey_in_mth5('/path/to/file.mth5', 'survey_01')
       >>> print(exists)
       True

   Check in version 0.2.0::

       >>> exists = survey_in_mth5('/path/to/file.mth5', survey_id='MT')
       >>> if exists:
       ...     print(f"Survey MT found in file")


