mth5.groups
===========

.. py:module:: mth5.groups

.. autoapi-nested-parse::

   Import all Group objects



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/groups/base/index
   /autoapi/mth5/groups/channel_dataset/index
   /autoapi/mth5/groups/estimate_dataset/index
   /autoapi/mth5/groups/experiment/index
   /autoapi/mth5/groups/fc_dataset/index
   /autoapi/mth5/groups/feature_dataset/index
   /autoapi/mth5/groups/features/index
   /autoapi/mth5/groups/filter_groups/index
   /autoapi/mth5/groups/filters/index
   /autoapi/mth5/groups/fourier_coefficients/index
   /autoapi/mth5/groups/reports/index
   /autoapi/mth5/groups/run/index
   /autoapi/mth5/groups/standards/index
   /autoapi/mth5/groups/station/index
   /autoapi/mth5/groups/survey/index
   /autoapi/mth5/groups/transfer_function/index


Classes
-------

.. autoapisummary::

   mth5.groups.BaseGroup
   mth5.groups.ReportsGroup
   mth5.groups.StandardsGroup
   mth5.groups.FiltersGroup
   mth5.groups.EstimateDataset
   mth5.groups.FCChannelDataset
   mth5.groups.MasterFCGroup
   mth5.groups.FCGroup
   mth5.groups.FCDecimationGroup
   mth5.groups.TransferFunctionsGroup
   mth5.groups.TransferFunctionGroup
   mth5.groups.RunGroup
   mth5.groups.FeatureChannelDataset
   mth5.groups.MasterFeaturesGroup
   mth5.groups.FeatureGroup
   mth5.groups.FeatureTSRunGroup
   mth5.groups.FeatureFCRunGroup
   mth5.groups.FeatureDecimationGroup
   mth5.groups.MasterStationGroup
   mth5.groups.StationGroup
   mth5.groups.MasterSurveyGroup
   mth5.groups.SurveyGroup
   mth5.groups.ExperimentGroup


Package Contents
----------------

.. py:class:: BaseGroup(group: h5py.Group | h5py.Dataset, group_metadata: mt_metadata.base.MetadataBase | None = None, **kwargs: Any)

   Base class for HDF5 group management with metadata handling.

   Provides core functionality for reading, writing, and managing HDF5 groups
   with integrated metadata validation using mt_metadata standards.

   :param group: HDF5 group or dataset object to wrap.
   :type group: h5py.Group or h5py.Dataset
   :param group_metadata: Metadata container with validated attributes. Default is None.
   :type group_metadata: MetadataBase, optional
   :param \*\*kwargs: Additional keyword arguments to set as instance attributes.
   :type \*\*kwargs: dict

   .. attribute:: hdf5_group

      Weak reference to the underlying HDF5 group.

      :type: h5py.Group or h5py.Dataset

   .. attribute:: metadata

      Metadata object with validation and standards compliance.

      :type: MetadataBase

   .. attribute:: logger

      Logger instance for tracking operations.

      :type: loguru.Logger

   .. attribute:: compression

      HDF5 compression method (e.g., 'gzip').

      :type: str, optional

   .. attribute:: compression_opts

      Compression options/level.

      :type: int, optional

   .. attribute:: shuffle

      Enable HDF5 shuffle filter. Default is False.

      :type: bool

   .. attribute:: fletcher32

      Enable HDF5 Fletcher32 checksum. Default is False.

      :type: bool

   .. rubric:: Notes

   - All HDF5 group references are weak references to prevent lingering
     file references after the group is closed.
   - Metadata changes should be written using `write_metadata()` method.
   - This is a base class inherited by more specific group types like
     SurveyGroup, StationGroup, RunGroup, etc.

   .. rubric:: Examples

   Create and manage a group with metadata

   >>> import h5py
   >>> with h5py.File('data.h5', 'r+') as f:
   ...     group = f.create_group('MyGroup')
   ...     base_obj = BaseGroup(group)
   ...     print(base_obj)
   ...     # Set and write metadata
   ...     base_obj.metadata.id = 'MyGroup'
   ...     base_obj.write_metadata()

   Access metadata and group structure

   >>> print(base_obj.metadata.id)
   'MyGroup'
   >>> print(base_obj.groups_list)
   ['subgroup1', 'subgroup2']
   >>> print(base_obj.hdf5_group.ref)  # Get HDF5 reference
   <HDF5 Group Reference>


   .. py:attribute:: compression
      :value: None



   .. py:attribute:: compression_opts
      :value: None



   .. py:attribute:: shuffle
      :value: False



   .. py:attribute:: fletcher32
      :value: False



   .. py:attribute:: logger


   .. py:property:: metadata
      :type: mt_metadata.base.MetadataBase


      Get metadata object with lazy loading from HDF5 attributes.

      :returns: Metadata container with all attributes and validation.
      :rtype: MetadataBase

      .. rubric:: Notes

      Metadata is loaded on first access and cached for subsequent accesses.

      .. rubric:: Examples

      >>> meta = base_obj.metadata
      >>> print(meta.id)
      'MyGroup'
      >>> print(meta.mth5_type)
      'Survey'


   .. py:property:: groups_list
      :type: list[str]


      Get list of all subgroup names in the HDF5 group.

      :returns: Names of all subgroups and datasets.
      :rtype: list of str

      .. rubric:: Examples

      >>> print(base_obj.groups_list)
      ['Station_001', 'Station_002', 'metadata']


   .. py:property:: dataset_options
      :type: dict[str, Any]


      Get the HDF5 dataset creation options.

      :returns: Dictionary containing compression, shuffle, and checksum settings.
      :rtype: dict

      .. rubric:: Examples

      >>> options = base_obj.dataset_options
      >>> print(options)
      {'compression': 'gzip', 'compression_opts': 4,
       'shuffle': True, 'fletcher32': False}


   .. py:method:: read_metadata() -> None

      Read metadata from HDF5 group attributes into metadata object.

      Loads all HDF5 attributes and converts them to appropriate Python types
      before populating the metadata object with validation.

      .. rubric:: Notes

      This method is called automatically on first metadata access if metadata
      has not been read yet. Empty attributes are skipped with a debug message.

      .. rubric:: Examples

      Manually read metadata after file changes

      >>> base_obj.read_metadata()
      >>> print(base_obj.metadata.id)
      'MyGroup'

      Check what attributes were read

      >>> base_obj.read_metadata()
      >>> attrs = list(base_obj.metadata.to_dict().keys())
      >>> print(f"Attributes: {attrs}")
      Attributes: ['id', 'comments', 'provenance']



   .. py:method:: write_metadata() -> None

      Write metadata from object to HDF5 group attributes.

      Converts metadata values to numpy-compatible types before writing to
      HDF5 attributes. Handles read-only mode gracefully with warnings.

      :raises KeyError: If HDF5 write fails for reasons other than read-only mode.
      :raises ValueError: If synchronous group creation fails for reasons other than read-only mode.

      .. rubric:: Notes

      - Keys that already exist are overwritten.
      - Read-only files will log a warning instead of raising an error.
      - This method should be called after any metadata changes.

      .. rubric:: Examples

      Update metadata and write to file

      >>> base_obj.metadata.id = 'UpdatedGroup'
      >>> base_obj.metadata.comments = 'New comments'
      >>> base_obj.write_metadata()

      Verify write by reloading

      >>> base_obj._has_read_metadata = False
      >>> base_obj.read_metadata()
      >>> print(base_obj.metadata.id)
      'UpdatedGroup'



   .. py:method:: initialize_group(**kwargs: Any) -> None

      Initialize group by setting attributes and writing metadata.

      Convenience method that sets keyword arguments as instance attributes
      and writes all metadata to the HDF5 file.

      :param \*\*kwargs: Key-value pairs to set as instance attributes.
      :type \*\*kwargs: dict

      .. rubric:: Examples

      Initialize with compression settings

      >>> base_obj.initialize_group(
      ...     compression='gzip',
      ...     compression_opts=4,
      ...     shuffle=True
      ... )



   .. py:method:: rename_group(new_name: str) -> None

      Rename the current group in the HDF5 file.

      :param new_name: New name for the group. Will be validated and normalized.
      :type new_name: str

      :raises MTH5Error: If renaming fails due to read-only mode or other issues.

      .. rubric:: Examples

      Rename a group

      >>> print(survey_obj.hdf5_group.name)
      '/OldSurveyName'
      >>> survey_obj.rename_group('NewSurveyName')
      >>> print(survey_obj.hdf5_group.name)
      '/NewSurveyName'



.. py:class:: ReportsGroup(group: h5py.Group, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Store report files (PDF/text) and images under ``/Survey/Reports``.

   Files are embedded into HDF5 datasets with basic metadata preserved.

   .. rubric:: Examples

   >>> reports = survey.reports_group
   >>> _ = reports.add_report("site_report", filename="/tmp/report.pdf")
   >>> _ = reports.get_report("site_report")  # doctest: +SKIP


   .. py:method:: add_report(report_name: str, report_metadata: dict[str, Any] | None = None, filename: str | pathlib.Path | None = None) -> None

      Add a report or image file to the group.

      :param report_name: Dataset name to store the file under.
      :type report_name: str
      :param report_metadata: Additional attributes to attach to the dataset.
      :type report_metadata: dict, optional
      :param filename: Path to the file to embed; supported types: PDF/TXT/MD and common images.
      :type filename: str or Path, optional

      :raises FileNotFoundError: If ``filename`` does not exist.

      .. rubric:: Examples

      >>> reports.add_report("manual", filename="docs/manual.pdf")  # doctest: +SKIP



   .. py:method:: get_report(report_name: str, write=True) -> pathlib.Path

      Extract a stored report or image to the current working directory.

      :param report_name: Name of the stored dataset.
      :type report_name: str

      :returns: Path to the materialized file on disk.
      :rtype: pathlib.Path

      :raises ValueError: If the stored file type is unsupported.

      .. rubric:: Examples

      >>> path = reports.get_report("site_report")  # doctest: +SKIP
      >>> path.exists()
      True



   .. py:method:: list_reports() -> list[str]

      List all stored reports and images in the group.

      :returns: Names of all stored datasets in the reports group.
      :rtype: list of str

      .. rubric:: Examples

      >>> report_names = reports.list_reports()  # doctest: +SKIP
      >>> print(report_names)
      ['site_report', 'manual', 'overview_image']



   .. py:method:: remove_report(report_name: str) -> None

      Remove a stored report or image from the group.

      :param report_name: Name of the stored dataset to remove.
      :type report_name: str

      .. rubric:: Examples

      >>> reports.remove_report("manual")  # doctest: +SKIP



.. py:class:: StandardsGroup(group: Any, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Container for metadata standards documentation stored in the HDF5 file.

   Stores metadata standards used throughout the survey in a standardized
   summary table. This enables users to understand metadata directly from
   the file without requiring external documentation.

   The standards are organized in a summary table at ``/Survey/Standards/summary``
   with columns for attribute name, type, requirements, style, units, and
   descriptions.

   .. attribute:: summary_table

      The standards summary table with metadata definitions.

      :type: MTH5Table

   .. rubric:: Notes

   Standards include definitions for:

   - Survey, Station, Run, Electric, Magnetic, Auxiliary metadata
   - Filter types: Coefficient, FIR, FrequencyResponseTable, PoleZero, TimeDelay
   - Processing standards from aurora and fourier_coefficients modules

   .. rubric:: Examples

   >>> with MTH5('survey.mth5') as mth5_obj:
   ...     standards = mth5_obj.standards_group
   ...     summary = standards.summary_table
   ...     print(summary.array.dtype.names)
   ('attribute', 'type', 'required', 'style', 'units', 'description', ...)

   Get information about a specific attribute:

   >>> standards.get_attribute_information('survey.release_license')
   survey.release_license
   --------------------------
           type          : string
           required      : True
           style         : controlled vocabulary
           ...


   .. py:property:: summary_table
      :type: mth5.tables.MTH5Table



   .. py:method:: get_attribute_information(attribute_name: str) -> None

      Print detailed information about a metadata attribute.

      Retrieves and displays all metadata standards information for
      the specified attribute from the standards summary table.

      :param attribute_name: Name of the attribute to describe (e.g., 'survey.release_license').
      :type attribute_name: str

      :raises MTH5TableError: If the attribute is not found in the standards summary table.

      .. rubric:: Notes

      Prints formatted output including:

      - Data type
      - Whether attribute is required
      - Style (e.g., controlled vocabulary)
      - Units
      - Description
      - Valid options
      - Aliases
      - Example values
      - Default value

      .. rubric:: Examples

      >>> standards = mth5_obj.standards_group
      >>> standards.get_attribute_information('survey.release_license')
      survey.release_license
      --------------------------
              type          : string
              required      : True
              style         : controlled vocabulary
              units         :
              description   : How the data can be used. The options are based on
                       Creative Commons licenses.
              options       : CC-0,CC-BY,CC-BY-SA,CC-BY-ND,CC-BY-NC-SA
              alias         :
              example       : CC-0
              default       : CC-0



   .. py:method:: summary_table_from_dict(summary_dict: dict[str, Any]) -> None

      Populate summary table from a dictionary of metadata standards.

      Converts a flattened dictionary of metadata standards into rows
      in the HDF5 summary table.

      :param summary_dict: Flattened dictionary of all metadata standards. Keys are
                           attribute names, values are dictionaries with type, required,
                           style, units, description, etc.
      :type summary_dict: dict[str, Any]

      .. rubric:: Notes

      Processes dictionary values:

      - Lists are converted to comma-separated strings
      - None values become empty strings
      - Bytes are decoded to UTF-8

      .. todo:: Adapt method to accept pandas.DataFrame as alternative input.

      .. rubric:: Examples

      >>> standards = StandardsGroup(group)
      >>> metadata = summarize_metadata_standards()
      >>> standards.summary_table_from_dict(metadata)



   .. py:method:: get_standards_summary(modules: Optional[list[str]] = None) -> numpy.ndarray

      Get standards for specified metadata modules.

      Retrieves and concatenates standards arrays from one or more
      metadata modules for inclusion in the standards table.

      :param modules: List of module names to include (e.g., 'timeseries', 'filters').
                      If None, uses default modules: common, timeseries, timeseries.filters,
                      transfer_functions.tf, features, features.weights, processing,
                      processing.fourier_coefficients, processing.aurora.
                      Default is None.
      :type modules: list[str], optional

      :returns: Concatenated numpy structured array containing standards for all
                requested modules with dtype matching STANDARDS_DTYPE.
      :rtype: np.ndarray

      .. rubric:: Examples

      >>> standards = StandardsGroup(group)
      >>> ts_standards = standards.get_standards_summary(['timeseries'])
      >>> print(ts_standards.shape)
      (45,)

      Get all default modules:

      >>> all_standards = standards.get_standards_summary()



   .. py:method:: summary_table_from_array(array: numpy.ndarray) -> None

      Populate summary table from a numpy structured array.

      Converts a structured numpy array into rows in the HDF5 summary table.

      :param array: Structured numpy array with dtype matching STANDARDS_DTYPE.
                    Each row represents one metadata attribute definition.
      :type array: np.ndarray

      .. rubric:: Notes

      Iterates through all rows of the structured array and adds them
      sequentially to the summary table using add_row().

      .. rubric:: Examples

      >>> standards = StandardsGroup(group)
      >>> standards_array = standards.get_standards_summary()
      >>> standards.summary_table_from_array(standards_array)



   .. py:method:: initialize_group() -> None

      Initialize the standards group and create the summary table.

      Creates the summary table dataset in the HDF5 file and populates it
      with metadata standards from all default modules. Sets appropriate
      HDF5 attributes and writes the group metadata.

      .. rubric:: Notes

      Initialization process:

      1. Creates HDF5 dataset for summary table with maximum expandable shape
      2. Applies compression if configured in dataset_options
      3. Sets HDF5 attributes: type, last_updated, reference
      4. Populates table with standards from all default modules
      5. Writes group metadata to HDF5

      The summary table uses STANDARDS_DTYPE and supports up to 1000 rows.

      .. rubric:: Examples

      >>> mth5_obj.initialize_group()
      >>> summary_table = mth5_obj.standards_group.summary_table
      >>> print(summary_table.array.shape)
      (342,)



.. py:class:: FiltersGroup(group: h5py.Group, **kwargs)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Container for managing all filter types in MTH5 format.

   This class provides a unified interface for organizing and accessing filters
   of different types. It automatically creates and manages subgroups for each
   filter type (ZPK, Coefficient, Time Delay, FAP, and FIR) within the HDF5
   file structure.

   Filter Types
   -----------
   - **zpk**: Zeros, Poles, and Gain representation
   - **coefficient**: FIR coefficient filter
   - **time_delay**: Time delay filter
   - **fap**: Frequency-Amplitude-Phase (FAP) lookup table
   - **fir**: Finite Impulse Response filter

   :param group: HDF5 group object for the filters container.
   :type group: h5py.Group
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. attribute:: zpk_group

      Subgroup for zeros-poles-gain filters.

      :type: ZPKGroup

   .. attribute:: coefficient_group

      Subgroup for coefficient filters.

      :type: CoefficientGroup

   .. attribute:: time_delay_group

      Subgroup for time delay filters.

      :type: TimeDelayGroup

   .. attribute:: fap_group

      Subgroup for frequency-amplitude-phase filters.

      :type: FAPGroup

   .. attribute:: fir_group

      Subgroup for FIR filters.

      :type: FIRGroup

   .. rubric:: Examples

   >>> import h5py
   >>> from mth5.groups.filters import FiltersGroup
   >>> with h5py.File('data.h5', 'r') as f:
   ...     filters = FiltersGroup(f['Filters'])
   ...     all_filters = filters.filter_dict
   ...     zpk_filter = filters.to_filter_object('my_zpk_filter')


   .. py:property:: filter_dict
      :type: dict[str, Any]


      Get a dictionary of all filters across all filter type groups.

      Aggregates filters from all subgroups (ZPK, Coefficient, Time Delay, FAP, FIR)
      into a single dictionary for convenient access and querying.

      :returns: Dictionary mapping filter names to filter metadata dictionaries.
                Each entry contains filter information including type and HDF5 reference.
      :rtype: dict[str, Any]

      .. rubric:: Examples

      >>> filters = FiltersGroup(h5_group)
      >>> all_filters = filters.filter_dict
      >>> print(list(all_filters.keys()))
      ['my_zpk_filter', 'lowpass_coefficient', 'time_delay_1', ...]
      >>> print(all_filters['my_zpk_filter']['type'])
      'zpk'


   .. py:method:: add_filter(filter_object: object) -> object

      Add a filter dataset based on its type.

      Automatically detects the filter type and routes the filter to the
      appropriate subgroup. Filter names are normalized to lowercase and
      forward slashes are replaced with " per " for consistency.

      :param filter_object: An MT metadata filter object with a 'type' attribute.
                            Supported types:

                            - 'zpk', 'poles_zeros': Zeros-Poles-Gain filter
                            - 'coefficient': Coefficient filter
                            - 'time_delay', 'time delay': Time delay filter
                            - 'fap', 'frequency response table': Frequency-Amplitude-Phase filter
                            - 'fir': Finite Impulse Response filter
      :type filter_object: mt_metadata.timeseries.filters

      :returns: Filter group object from the appropriate subgroup.
      :rtype: object

      .. rubric:: Notes

      If a filter with the same name already exists, the existing filter
      is returned instead of creating a duplicate.

      .. rubric:: Examples

      >>> from mt_metadata.timeseries.filters import ZPK
      >>> filters = FiltersGroup(h5_group)
      >>> zpk_filter = ZPK(name='my_filter')
      >>> added_filter = filters.add_filter(zpk_filter)

      Add coefficient filter:

      >>> from mt_metadata.timeseries.filters import Coefficient
      >>> coeff_filter = Coefficient(name='lowpass')
      >>> filters.add_filter(coeff_filter)



   .. py:method:: get_filter(name: str) -> h5py.Dataset | h5py.Group

      Retrieve a filter dataset by name.

      Looks up the filter by name in the aggregated filter dictionary and
      returns the HDF5 dataset or group object.

      :param name: Name of the filter to retrieve.
      :type name: str

      :returns: HDF5 dataset or group object for the requested filter.
      :rtype: h5py.Dataset or h5py.Group

      :raises KeyError: If the filter name is not found in the filter dictionary.

      .. rubric:: Examples

      >>> filters = FiltersGroup(h5_group)
      >>> filter_dataset = filters.get_filter('my_zpk_filter')
      >>> print(filter_dataset.attrs)



   .. py:method:: to_filter_object(name: str) -> object

      Convert a filter HDF5 dataset to an MT metadata filter object.

      Retrieves the filter metadata from the HDF5 file and converts it to
      the appropriate MT metadata filter class based on filter type.

      :param name: Name of the filter to convert.
      :type name: str

      :returns: MT metadata filter object (ZPK, Coefficient, TimeDelay, FAP, or FIR).
      :rtype: object

      :raises KeyError: If the filter name is not found in the filter dictionary.

      .. rubric:: Examples

      >>> filters = FiltersGroup(h5_group)
      >>> zpk_filter = filters.to_filter_object('my_zpk_filter')
      >>> print(zpk_filter.name)
      'my_zpk_filter'
      >>> print(type(zpk_filter))
      <class 'mt_metadata.timeseries.filters.ZPK'>

      Get different filter types:

      >>> coeff_filter = filters.to_filter_object('lowpass_coefficient')
      >>> fap_filter = filters.to_filter_object('frequency_response_1')



.. py:class:: EstimateDataset(dataset: h5py.Dataset, dataset_metadata: mt_metadata.transfer_functions.tf.statistical_estimate.StatisticalEstimate | None = None, write_metadata: bool = True, **kwargs: Any)

   Container for statistical estimates of transfer functions.

   This class holds multi-dimensional statistical estimates for transfer
   functions with full metadata management. Estimates are stored as HDF5
   datasets with dimensions for period, output channels, and input channels.

   :param dataset: HDF5 dataset containing the statistical estimate data.
   :type dataset: h5py.Dataset
   :param dataset_metadata: Metadata object for the estimate. If provided and write_metadata is True,
                            the metadata will be written to the HDF5 attributes. Defaults to None.
   :type dataset_metadata: mt_metadata.transfer_functions.tf.StatisticalEstimate, optional
   :param write_metadata: If True, write metadata to the HDF5 dataset attributes. Defaults to True.
   :type write_metadata: bool, optional
   :param \*\*kwargs: Additional keyword arguments (reserved for future use).
   :type \*\*kwargs: Any

   .. attribute:: hdf5_dataset

      Weak reference to the HDF5 dataset.

      :type: h5py.Dataset

   .. attribute:: metadata

      Metadata container for the estimate.

      :type: StatisticalEstimate

   .. attribute:: logger

      Logger instance for reporting messages.

      :type: loguru.logger

   :raises MTH5Error: If dataset_metadata is provided but is not of type StatisticalEstimate
       or a compatible metadata class.
   :raises TypeError: If input data cannot be converted to numpy array or has wrong dtype/shape.

   .. rubric:: Notes

   The estimate data is stored in 3D form with shape:
   (n_periods, n_output_channels, n_input_channels)

   Metadata is automatically synchronized between the pydantic model and
   HDF5 attributes on initialization and after any modifications.

   .. rubric:: Examples

   Create an estimate dataset from an HDF5 group:

   >>> import h5py
   >>> import numpy as np
   >>> from mt_metadata.transfer_functions.tf.statistical_estimate import StatisticalEstimate
   >>> # Create HDF5 file with estimate dataset
   >>> with h5py.File('estimate.h5', 'w') as f:
   ...     # Create dataset with shape (10 periods, 2 outputs, 2 inputs)
   ...     data = np.random.rand(10, 2, 2)
   ...     dset = f.create_dataset('estimate', data=data)
   ...     # Create EstimateDataset
   ...     est = EstimateDataset(dset, write_metadata=True)

   Convert estimate to xarray and back:

   >>> periods = np.logspace(-3, 3, 10)  # 10 periods from 1e-3 to 1e3 s
   >>> xr_data = est.to_xarray(periods)
   >>> # Modify xarray coordinates
   >>> new_xr = xr_data.rename({'output': 'new_output', 'input': 'new_input'})
   >>> est.from_xarray(new_xr)  # Load modified data back

   Access estimate data in different formats:

   >>> # Get numpy array
   >>> np_data = est.to_numpy()
   >>> print(np_data.shape)  # (10, 2, 2)
   >>> # Get xarray with proper coordinates
   >>> xr_data = est.to_xarray(periods)
   >>> print(xr_data.dims)  # ('period', 'output', 'input')


   .. py:attribute:: logger


   .. py:attribute:: metadata


   .. py:method:: read_metadata() -> None

      Read metadata from HDF5 attributes into metadata container.

      Reads all attributes from the HDF5 dataset and loads them into
      the internal metadata object for validation and access.

      :rtype: None

      .. rubric:: Notes

      This is automatically called during initialization if 'mth5_type'
      attribute exists in the HDF5 dataset.

      .. rubric:: Examples

      Reload metadata from HDF5 after external modification:

      >>> # Metadata was modified in HDF5
      >>> est.read_metadata()  # Reload changes
      >>> print(est.metadata.name)  # Access updated name



   .. py:method:: write_metadata() -> None

      Write metadata from container to HDF5 dataset attributes.

      Converts the pydantic metadata model to a dictionary and writes
      each field as an HDF5 attribute. Values are converted to appropriate
      numpy types for compatibility.

      :rtype: None

      .. rubric:: Notes

      All existing attributes with the same names will be overwritten.
      This is called automatically during initialization and after
      metadata updates.

      .. rubric:: Examples

      Save updated metadata to HDF5:

      >>> est.metadata.name = "Updated Estimate"
      >>> est.write_metadata()  # Persist to file
      >>> # Verify write
      >>> print(est.hdf5_dataset.attrs['name'])
      b'Updated Estimate'



   .. py:method:: replace_dataset(new_data_array: numpy.ndarray) -> None

      Replace entire dataset with new data.

      Resizes the HDF5 dataset if necessary and replaces all data.
      Converts input to numpy array if needed.

      :param new_data_array: New estimate data to store. Should have shape
                             (n_periods, n_output_channels, n_input_channels).
      :type new_data_array: np.ndarray

      :rtype: None

      :raises TypeError: If input cannot be converted to numpy array.

      .. rubric:: Notes

      If new data has different shape, HDF5 dataset will be resized.
      This is generally safe but may fragment the HDF5 file.

      .. rubric:: Examples

      Replace estimate with new data:

      >>> import numpy as np
      >>> new_estimate = np.random.rand(10, 2, 2)  # 10 periods, 2 channels
      >>> est.replace_dataset(new_estimate)
      >>> print(est.to_numpy().shape)
      (10, 2, 2)

      Replace with data from list (auto-converted to array):

      >>> data_list = [[[1, 2], [3, 4]]] * 5  # 5 periods
      >>> est.replace_dataset(data_list)
      >>> est.to_numpy().shape
      (5, 2, 2)



   .. py:method:: to_xarray(period: numpy.ndarray | list) -> xarray.DataArray

      Convert estimate to xarray DataArray.

      Creates an xarray DataArray with proper coordinates for periods,
      output channels, and input channels. Includes metadata as attributes.

      :param period: Period values for coordinate. Should have length equal to
                     estimate first dimension (n_periods).
      :type period: np.ndarray | list

      :returns: DataArray with dimensions (period, output, input) and
                coordinates from metadata.
      :rtype: xr.DataArray

      .. rubric:: Notes

      Metadata changes in xarray are not validated and will not be
      synchronized back to HDF5 without explicit call to from_xarray().
      Data is loaded entirely into memory.

      .. rubric:: Examples

      Convert to xarray with logarithmic period spacing:

      >>> import numpy as np
      >>> periods = np.logspace(-2, 3, 10)  # 10 periods from 0.01 to 1000
      >>> xr_data = est.to_xarray(periods)
      >>> print(xr_data.dims)
      ('period', 'output', 'input')
      >>> print(xr_data.coords['period'].values)
      [1.00e-02 3.16e-02 ... 1.00e+03]

      Select data by period range:

      >>> subset = xr_data.sel(period=slice(0.1, 100))
      >>> print(subset.shape)
      (8, 2, 2)



   .. py:method:: to_numpy() -> numpy.ndarray

      Convert estimate to numpy array.

      Returns the HDF5 dataset as a numpy array. Data is loaded
      entirely into memory.

      :returns: 3D array with shape (n_periods, n_output_channels, n_input_channels).
      :rtype: np.ndarray

      .. rubric:: Notes

      For large estimates, this loads all data into RAM. Consider using
      HDF5 slicing for memory-efficient access.

      .. rubric:: Examples

      Get full estimate as numpy array:

      >>> data = est.to_numpy()
      >>> print(data.shape)
      (10, 2, 2)
      >>> print(data.dtype)
      float64

      Access specific period and channels:

      >>> data = est.to_numpy()
      >>> # Get first 5 periods, output channel 0, input channel 1
      >>> subset = data[:5, 0, 1]
      >>> print(subset.shape)
      (5,)



   .. py:method:: from_numpy(new_estimate: numpy.ndarray) -> None

      Load estimate data from numpy array.

      Validates dtype and shape compatibility, resizes dataset if needed,
      and stores the data.

      :param new_estimate: Estimate data to load. Must be convertible to numpy array.
                           Preferred shape: (n_periods, n_output_channels, n_input_channels).
      :type new_estimate: np.ndarray

      :rtype: None

      :raises TypeError: If dtype doesn't match existing dataset or input cannot
          be converted to numpy array.

      .. rubric:: Notes

      'data' is a built-in Python function and cannot be used as parameter name.
      The dataset will be resized if shape doesn't match.

      .. rubric:: Examples

      Load estimate from numpy array:

      >>> import numpy as np
      >>> new_data = np.random.rand(5, 2, 2)
      >>> est.from_numpy(new_data)
      >>> print(est.to_numpy().shape)
      (5, 2, 2)

      Load with automatic dtype conversion:

      >>> float_data = np.array([[[1.0, 2.0]]], dtype=np.float64)
      >>> est.from_numpy(float_data)



   .. py:method:: from_xarray(data: xarray.DataArray) -> None

      Load estimate data from xarray DataArray.

      Updates metadata from xarray coordinates and attributes, then
      stores the data.

      :param data: DataArray containing estimate. Expected dimensions:
                   (period, output, input).
      :type data: xr.DataArray

      :rtype: None

      .. rubric:: Notes

      This will update output_channels, input_channels, name, and data_type
      from the xarray object. All changes are persisted to HDF5.

      .. rubric:: Examples

      Load estimate from modified xarray:

      >>> xr_data = est.to_xarray(periods)
      >>> # Modify data and metadata
      >>> modified = xr_data * 2  # Scale by 2
      >>> est.from_xarray(modified)
      >>> print(est.to_numpy()[0, 0, 0])  # Verify scale

      Rename channels and reload:

      >>> xr_data = est.to_xarray(periods)
      >>> new_xr = xr_data.rename({
      ...     'output': ['Ex', 'Ey'],
      ...     'input': ['Bx', 'By']
      ... })
      >>> est.from_xarray(new_xr)
      >>> print(est.metadata.output_channels)
      ['Ex', 'Ey']



.. py:class:: FCChannelDataset(dataset: h5py.Dataset, dataset_metadata: mt_metadata.processing.fourier_coefficients.FCChannel | None = None, **kwargs: Any)

   Container for Fourier coefficients (FC) from windowed FFT analysis.

   Holds multi-dimensional Fourier coefficient data representing time-frequency
   analysis results. Data is uniformly sampled in both frequency (via harmonic
   index) and time (via uniform FFT window step size).

   :param dataset: HDF5 dataset containing the Fourier coefficient data.
   :type dataset: h5py.Dataset
   :param dataset_metadata: Metadata object containing FC channel properties like start time,
                            end time, sample rates, units, and frequency method. If provided,
                            metadata will be written to HDF5 attributes. Defaults to None.
   :type dataset_metadata: FCChannel | None, optional
   :param \*\*kwargs: Additional keyword arguments (reserved for future use).
   :type \*\*kwargs: Any

   .. attribute:: hdf5_dataset

      Weak reference to the HDF5 dataset.

      :type: h5py.Dataset

   .. attribute:: metadata

      Metadata container for the Fourier coefficients.

      :type: FCChannel

   .. attribute:: logger

      Logger instance for reporting messages.

      :type: loguru.logger

   :raises MTH5Error: If dataset_metadata is provided but is not of type FCChannel.
   :raises TypeError: If input data cannot be converted to numpy array or has
       incompatible dtype/shape.

   .. rubric:: Notes

   The data array has shape (n_windows, n_frequencies) where:
   - n_windows: Number of time windows in the FFT moving window analysis
   - n_frequencies: Number of frequency bins determined by window size

   Data is typically complex-valued representing Fourier coefficients.
   Time windows are uniformly spaced with interval 1/sample_rate_window_step.
   Frequencies are uniformly spaced from frequency_min to frequency_max.

   Metadata includes:
   - Time period (start and end)
   - Acquisition and decimated sample rates
   - Window sample rate (delta_t within window)
   - Units
   - Frequency method (integer harmonic index calculation)
   - Component name (channel designation)

   .. rubric:: Examples

   Create an FC dataset from HDF5 group:

   >>> import h5py
   >>> import numpy as np
   >>> from mt_metadata.processing.fourier_coefficients import FCChannel
   >>> with h5py.File('fc.h5', 'w') as f:
   ...     # Create 2D array: 50 time windows, 256 frequencies
   ...     data = np.random.rand(50, 256) + 1j * np.random.rand(50, 256)
   ...     dset = f.create_dataset('Ex', data=data, dtype=np.complex128)
   ...     # Create FCChannelDataset
   ...     fc = FCChannelDataset(dset, write_metadata=True)

   Convert to xarray and access time-frequency data:

   >>> xr_data = fc.to_xarray()
   >>> print(xr_data.dims)  # ('time', 'frequency')
   >>> # Access data at specific time and frequency
   >>> subset = xr_data.sel(time='2023-01-01T12:00:00', method='nearest')

   Inspect properties:

   >>> print(f"Windows: {fc.n_windows}, Frequencies: {fc.n_frequencies}")
   >>> print(f"Frequency range: {fc.frequency.min():.2f}-{fc.frequency.max():.2f} Hz")


   .. py:attribute:: logger


   .. py:attribute:: metadata


   .. py:method:: read_metadata() -> None

      Read metadata from HDF5 attributes into metadata container.

      Reads all attributes from the HDF5 dataset and loads them into
      the internal metadata object for validation and access.

      :rtype: None

      .. rubric:: Notes

      This is automatically called during initialization if 'mth5_type'
      attribute exists in the HDF5 dataset.

      .. rubric:: Examples

      Reload metadata from HDF5 after external modification:

      >>> # Metadata was modified in HDF5
      >>> fc.read_metadata()  # Reload changes
      >>> print(fc.metadata.component)  # Access updated component



   .. py:method:: write_metadata() -> None

      Write metadata from container to HDF5 dataset attributes.

      Converts the pydantic metadata model to a dictionary and writes
      each field as an HDF5 attribute. Values are converted to appropriate
      numpy types for compatibility. Always ensures 'mth5_type' attribute
      is set to 'FCChannel'.

      :rtype: None

      .. rubric:: Notes

      All existing attributes with the same names will be overwritten.
      This is called automatically during initialization and after
      metadata updates. Read-only files will silently skip writes.

      .. rubric:: Examples

      Save updated metadata to HDF5:

      >>> fc.metadata.component = "Ey"
      >>> fc.write_metadata()  # Persist to file
      >>> # Verify write
      >>> print(fc.hdf5_dataset.attrs['component'])
      b'Ey'



   .. py:property:: n_windows
      :type: int


      Number of time windows in the FFT analysis.

      :returns: Number of time windows (first dimension of data array).
      :rtype: int

      .. rubric:: Notes

      This corresponds to the number of rows in the 2D spectrogram data.
      Each window represents a uniform time interval determined by the
      window step size (1/sample_rate_window_step).

      .. rubric:: Examples

      >>> print(f"Time windows: {fc.n_windows}")
      Time windows: 50


   .. py:property:: time
      :type: numpy.ndarray


      Time array including the start of each time window.

      Generates uniformly spaced time coordinates based on the start time,
      window step rate, and number of windows. Uses metadata time period
      to determine bounds.

      :returns: Array of datetime64 values for each window start time.
      :rtype: np.ndarray

      .. rubric:: Notes

      Time coordinates are generated using make_dt_coordinates, which
      ensures consistency between specified start/end times and the
      number of windows.

      .. rubric:: Examples

      Access time array for time-based indexing:

      >>> time_array = fc.time
      >>> print(time_array.shape)  # (n_windows,)
      >>> print(time_array[0])  # First window time
      2023-01-01T00:00:00.000000


   .. py:property:: n_frequencies
      :type: int


      Number of frequency bins in the Fourier analysis.

      :returns: Number of frequency bins (second dimension of data array).
      :rtype: int

      .. rubric:: Notes

      This corresponds to the number of columns in the 2D spectrogram data.
      Determined by the FFT window size and relates to the frequency
      resolution of the analysis.

      .. rubric:: Examples

      >>> print(f"Frequency bins: {fc.n_frequencies}")
      Frequency bins: 256


   .. py:property:: frequency
      :type: numpy.ndarray


      Frequency array from metadata frequency bounds.

      Generates uniformly spaced frequency coordinates based on the
      metadata frequency range and number of frequency bins.

      :returns: Array of frequency values, linearly spaced from frequency_min
                to frequency_max.
      :rtype: np.ndarray

      .. rubric:: Notes

      Frequencies represent harmonic indices or actual frequency values
      depending on the frequency method specified in metadata.
      Spacing is determined by n_frequencies bins over the range.

      .. rubric:: Examples

      Access frequency array for frequency-based indexing:

      >>> freq_array = fc.frequency
      >>> print(freq_array.shape)  # (n_frequencies,)
      >>> print(f"Frequency range: {freq_array.min():.2f} to {freq_array.max():.2f} Hz")
      Frequency range: 0.00 to 64.00 Hz


   .. py:method:: replace_dataset(new_data_array: numpy.ndarray) -> None

      Replace entire dataset with new data.

      Resizes the HDF5 dataset if necessary and replaces all data.
      Converts input to numpy array if needed.

      :param new_data_array: New FC data to store. Should have shape (n_windows, n_frequencies)
                             and typically complex-valued.
      :type new_data_array: np.ndarray

      :rtype: None

      :raises TypeError: If input cannot be converted to numpy array.

      .. rubric:: Notes

      If new data has different shape, HDF5 dataset will be resized.
      This is generally safe but may fragment the HDF5 file.

      .. rubric:: Examples

      Replace FC data with new analysis results:

      >>> import numpy as np
      >>> new_fc = np.random.rand(30, 256) + 1j * np.random.rand(30, 256)
      >>> fc.replace_dataset(new_fc)
      >>> print(fc.to_numpy().shape)
      (30, 256)

      Replace with data from list (auto-converted to array):

      >>> data_list = [[[1+1j, 2+2j]], [[3+3j, 4+4j]]] * 15
      >>> fc.replace_dataset(data_list)
      >>> fc.to_numpy().shape
      (30, 2)



   .. py:method:: to_xarray() -> xarray.DataArray

      Convert FC data to xarray DataArray.

      Creates an xarray DataArray with proper coordinates for time and
      frequency. Includes metadata as attributes.

      :returns: DataArray with dimensions (time, frequency) and coordinates
                from metadata and computed properties.
      :rtype: xr.DataArray

      .. rubric:: Notes

      Metadata changes in xarray are not validated and will not be
      synchronized back to HDF5 without explicit call to from_xarray().
      Data is loaded entirely into memory.

      .. rubric:: Examples

      Convert to xarray with automatic coordinates:

      >>> xr_data = fc.to_xarray()
      >>> print(xr_data.dims)
      ('time', 'frequency')
      >>> print(xr_data.shape)
      (50, 256)

      Select data by time and frequency range:

      >>> subset = xr_data.sel(
      ...     time=slice('2023-01-01T00:00:00', '2023-01-01T12:00:00'),
      ...     frequency=slice(0, 10)
      ... )
      >>> print(subset.shape)  # Subset shape



   .. py:method:: to_numpy() -> numpy.ndarray

      Convert FC data to numpy array.

      Returns the HDF5 dataset as a numpy array. Data is loaded
      entirely into memory.

      :returns: 2D complex array with shape (n_windows, n_frequencies).
      :rtype: np.ndarray

      .. rubric:: Notes

      For large spectrograms, this loads all data into RAM. Consider using
      HDF5 slicing for memory-efficient access to subsets.

      .. rubric:: Examples

      Get full FC data as numpy array:

      >>> data = fc.to_numpy()
      >>> print(data.shape)
      (50, 256)
      >>> print(data.dtype)
      complex128

      Access specific time window and frequency:

      >>> data = fc.to_numpy()
      >>> # Get first 10 windows, frequency bin 100
      >>> subset = data[:10, 100]
      >>> print(subset.shape)
      (10,)



   .. py:method:: from_numpy(new_estimate: numpy.ndarray) -> None

      Load FC data from numpy array.

      Validates dtype and shape compatibility, resizes dataset if needed,
      and stores the data.

      :param new_estimate: FC data to load. Should have shape (n_windows, n_frequencies).
                           Typically complex-valued array.
      :type new_estimate: np.ndarray

      :rtype: None

      :raises TypeError: If dtype doesn't match existing dataset or input cannot
          be converted to numpy array.

      .. rubric:: Notes

      'data' is a built-in Python function and cannot be used as parameter name.
      The dataset will be resized if shape doesn't match.
      Dtype compatibility is strictly enforced.

      .. rubric:: Examples

      Load FC data from numpy array:

      >>> import numpy as np
      >>> new_data = np.random.rand(25, 128) + 1j * np.random.rand(25, 128)
      >>> fc.from_numpy(new_data)
      >>> print(fc.to_numpy().shape)
      (25, 128)

      Load with magnitude and phase separation:

      >>> magnitude = np.random.rand(20, 256)
      >>> phase = np.random.rand(20, 256) * 2 * np.pi
      >>> fc_data = magnitude * np.exp(1j * phase)
      >>> fc.from_numpy(fc_data)



   .. py:method:: from_xarray(data: xarray.DataArray, sample_rate_decimation_level: int | float) -> None

      Load FC data from xarray DataArray.

      Updates metadata from xarray coordinates and attributes, then
      stores the data. Computes frequency and time parameters from
      the provided xarray object.

      :param data: DataArray containing FC data. Expected dimensions:
                   (time, frequency).
      :type data: xr.DataArray
      :param sample_rate_decimation_level: Decimation level applied to original sample rate.
                                           Used to track processing history.
      :type sample_rate_decimation_level: int | float

      :rtype: None

      .. rubric:: Notes

      This will update time_period (start/end), frequency bounds,
      window step rate, decimation level, component name, and units
      from the xarray object. All changes are persisted to HDF5.

      .. rubric:: Examples

      Load FC data from modified xarray:

      >>> xr_data = fc.to_xarray()
      >>> # Modify data (e.g., apply filter)
      >>> modified = xr_data * np.hamming(256)  # Apply frequency window
      >>> fc.from_xarray(modified, sample_rate_decimation_level=4)
      >>> print(fc.metadata.sample_rate_decimation_level)
      4

      Load with updated metadata from another analysis:

      >>> import xarray as xr
      >>> import pandas as pd
      >>> time_coords = pd.date_range('2023-01-01', periods=30, freq='1H')
      >>> freq_coords = np.arange(0, 128)
      >>> new_fc = xr.DataArray(
      ...     data=np.random.rand(30, 128) + 1j * np.random.rand(30, 128),
      ...     coords={'time': time_coords, 'frequency': freq_coords},
      ...     dims=['time', 'frequency'],
      ...     name='Ey',
      ...     attrs={'units': 'mV/km'}
      ... )
      >>> fc.from_xarray(new_fc, sample_rate_decimation_level=1)
      >>> print(fc.metadata.component)
      Ey



.. py:class:: MasterFCGroup(group: h5py.Group, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Master container for all Fourier Coefficient estimations of time series data.

   This class manages multiple Fourier Coefficient processing runs, each containing
   different decimation levels. No metadata is required at the master level.

   Hierarchy
   ---------
   MasterFCGroup -> FCGroup (processing runs) -> FCDecimationGroup (decimation levels)
   -> FCChannelDataset (individual channels)

   :param group: HDF5 group object for the master FC container.
   :type group: h5py.Group
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> import h5py
   >>> from mth5.groups.fourier_coefficients import MasterFCGroup
   >>> with h5py.File('data.h5', 'r') as f:
   ...     master = MasterFCGroup(f['FC'])
   ...     fc_group = master.add_fc_group('processing_run_1')


   .. py:property:: fc_summary
      :type: pandas.DataFrame


      Get a summary of all Fourier Coefficient processing runs.

      :returns: Summary information for all FC groups including names and metadata.
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> master = MasterFCGroup(h5_group)
      >>> summary = master.fc_summary


   .. py:method:: add_fc_group(fc_name: str, fc_metadata: Optional[mt_metadata.processing.fourier_coefficients.Decimation] = None) -> FCGroup

      Add a Fourier Coefficient processing run group.

      :param fc_name: Name for the FC group (usually identifies the processing run).
      :type fc_name: str
      :param fc_metadata: Metadata for the FC group. Default is None.
      :type fc_metadata: fc.Decimation, optional

      :returns: Newly created Fourier Coefficient group.
      :rtype: FCGroup

      .. rubric:: Examples

      >>> master = MasterFCGroup(h5_group)
      >>> fc_group = master.add_fc_group('processing_run_1')
      >>> print(fc_group.name)
      'processing_run_1'



   .. py:method:: get_fc_group(fc_name: str) -> FCGroup

      Retrieve a Fourier Coefficient group by name.

      :param fc_name: Name of the FC group to retrieve.
      :type fc_name: str

      :returns: The requested Fourier Coefficient group.
      :rtype: FCGroup

      :raises MTH5Error: If the FC group does not exist.

      .. rubric:: Examples

      >>> master = MasterFCGroup(h5_group)
      >>> fc_group = master.get_fc_group('processing_run_1')



   .. py:method:: remove_fc_group(fc_name: str) -> None

      Remove a Fourier Coefficient group.

      Deletes the specified FC group and all associated decimation levels and channels.

      :param fc_name: Name of the FC group to remove.
      :type fc_name: str

      :raises MTH5Error: If the FC group does not exist.

      .. rubric:: Examples

      >>> master = MasterFCGroup(h5_group)
      >>> master.remove_fc_group('processing_run_1')



.. py:class:: FCGroup(group: h5py.Group, decimation_level_metadata: Optional[mt_metadata.processing.fourier_coefficients.Decimation] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Manage a set of Fourier Coefficients from a single processing run.

   Holds Fourier Coefficient estimations organized by decimation level.
   Each decimation level contains channels (Ex, Ey, Hz, etc.) with complex
   frequency or time-frequency representations of the input signal.

   All channels must use the same calibration. Recalibration requires
   rerunning the Fourier Coefficient estimation.

   .. attribute:: hdf5_group

      The HDF5 group containing decimation levels

      :type: h5py.Group

   .. attribute:: metadata

      Decimation metadata including time period, sample rates, and channels

      :type: fc.Decimation

   .. rubric:: Notes

   Processing run structure:

   - Multiple decimation levels at different sample rates
   - Each decimation level contains multiple channels
   - Each channel contains complex Fourier coefficients
   - Time period and sample rates define the estimation window

   .. rubric:: Examples

   >>> with h5py.File('data.h5', 'r') as f:
   ...     fc_run = FCGroup(f['Fourier_Coefficients/run_1'])
   ...     print(fc_run.decimation_level_summary)


   .. py:method:: metadata() -> mt_metadata.processing.fourier_coefficients.Decimation

      Get processing run metadata including all decimation levels.

      Collects metadata from all decimation level groups and aggregates
      into a single Decimation metadata object.

      :returns: Metadata containing time period, sample rates, and all decimation
                level information.
      :rtype: fc.Decimation

      .. rubric:: Notes

      This getter automatically populates:

      - Time period (start and end)
      - List of all decimation levels and their metadata
      - HDF5 reference to this group

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> metadata = fc_run.metadata
      >>> print(metadata.time_period.start)
      2023-01-01T00:00:00



   .. py:property:: decimation_level_summary
      :type: pandas.DataFrame


      Get a summary of all decimation levels in this processing run.

      Returns information about each decimation level including sample rate,
      decimation level value, and time span.

      :returns: Summary with columns:

                - decimation_level: Integer decimation level identifier
                - start: ISO format start time of this decimation level
                - end: ISO format end time of this decimation level
                - hdf5_reference: Reference to the HDF5 group
      :rtype: pd.DataFrame

      .. rubric:: Notes

      Each row represents a single decimation level containing multiple
      channels with Fourier coefficients at different sample rates.

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> summary = fc_run.decimation_level_summary
      >>> print(summary[['decimation_level', 'start', 'end']])
         decimation_level                start                  end
      0              0     2023-01-01T00:00:00.000000  2023-01-01T01:00:00.000000
      1              1     2023-01-01T00:00:00.000000  2023-01-01T02:00:00.000000


   .. py:method:: add_decimation_level(decimation_level_name: str, decimation_level_metadata: Optional[dict | mt_metadata.processing.fourier_coefficients.Decimation] = None) -> FCDecimationGroup

      Add a new decimation level to the processing run.

      Creates a new FCDecimationGroup for a single decimation level containing
      Fourier Coefficient channels at a specific sample rate.

      :param decimation_level_name: Identifier for the decimation level.
      :type decimation_level_name: str
      :param decimation_level_metadata: Metadata for the decimation level. Can be a dictionary or
                                        fc.Decimation object. Default is None.
      :type decimation_level_metadata: dict | fc.Decimation, optional

      :returns: Newly created decimation level group.
      :rtype: FCDecimationGroup

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> metadata = fc.Decimation(decimation_level=0)
      >>> decimation = fc_run.add_decimation_level('0', metadata)



   .. py:method:: get_decimation_level(decimation_level_name: str) -> FCDecimationGroup

      Retrieve a decimation level by name.

      :param decimation_level_name: Name or identifier of the decimation level.
      :type decimation_level_name: str

      :returns: The requested decimation level group.
      :rtype: FCDecimationGroup

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> decimation = fc_run.get_decimation_level('0')
      >>> channels = decimation.groups_list



   .. py:method:: remove_decimation_level(decimation_level_name: str) -> None

      Remove a decimation level from the processing run.

      Deletes the HDF5 group and all its channels (FCChannelDataset objects).

      :param decimation_level_name: Name or identifier of the decimation level to remove.
      :type decimation_level_name: str

      .. rubric:: Notes

      This removes the entire decimation level and all channels within it.
      To remove individual channels, use FCDecimationGroup.remove_channel()
      instead.

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> fc_run.remove_decimation_level('0')



   .. py:method:: update_metadata() -> None

      Update processing run metadata from all decimation levels.

      Aggregates time period information from all decimation levels
      and writes updated metadata to HDF5.

      .. rubric:: Notes

      Collects:

      - Earliest start time across all decimation levels
      - Latest end time across all decimation levels

      Should be called after adding or removing decimation levels.

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> fc_run.add_decimation_level('0', metadata0)
      >>> fc_run.add_decimation_level('1', metadata1)
      >>> fc_run.update_metadata()



   .. py:method:: supports_aurora_processing_config(processing_config: aurora.config.metadata.processing.Processing, remote: bool) -> bool

      Check if all required decimation levels exist for Aurora processing.

      Performs an all-or-nothing check: returns True only if every decimation
      level required by the processing config is available in this FCGroup.

      Uses sequential logic to short-circuit: if any required decimation level
      is missing, immediately returns False without checking remaining levels.

      :param processing_config: Aurora processing configuration containing required decimation levels.
      :type processing_config: aurora.config.metadata.processing.Processing
      :param remote: Whether to check for remote processing compatibility.
      :type remote: bool

      :returns: True if all required decimation levels are available and consistent,
                False otherwise.
      :rtype: bool

      .. rubric:: Notes

      Validation logic:

      1. Extract list of decimation levels from processing config
      2. Iterate through each required level in sequence
      3. For each level, find a matching FCDecimation in this group
      4. Check consistency using Aurora's validation method
      5. If any level is missing or inconsistent, return False immediately
      6. Return True only if all levels pass validation

      .. rubric:: Examples

      >>> fc_run = FCGroup(h5_group)
      >>> config = aurora.config.metadata.processing.Processing(...)
      >>> if fc_run.supports_aurora_processing_config(config, remote=False):
      ...     # All decimation levels are available
      ...     pass



.. py:class:: FCDecimationGroup(group: h5py.Group, decimation_level_metadata: Optional[mt_metadata.processing.fourier_coefficients.Decimation] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for a single decimation level of Fourier Coefficient data.

   This class manages all channels at a specific decimation level, assuming
   uniform sampling in both frequency and time domains.

   Data Assumptions
   ----------------
   1. Data uniformly sampled in frequency domain
   2. Data uniformly sampled in time domain
   3. FFT moving window has uniform step size

   .. attribute:: start_time

      Start time of the decimation level

      :type: datetime

   .. attribute:: end_time

      End time of the decimation level

      :type: datetime

   .. attribute:: channels

      List of channel names in this decimation level

      :type: list

   .. attribute:: decimation_factor

      Factor by which data was decimated

      :type: int

   .. attribute:: decimation_level

      Level index in decimation hierarchy

      :type: int

   .. attribute:: sample_rate

      Sample rate after decimation (Hz)

      :type: float

   .. attribute:: method

      Method used (FFT, wavelet, etc.)

      :type: str

   .. attribute:: window

      Window parameters (length, overlap, type, sample rate)

      :type: dict

   :param group: HDF5 group object for this decimation level.
   :type group: h5py.Group
   :param decimation_level_metadata: Metadata for the decimation level. Default is None.
   :type decimation_level_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> decimation = FCDecimationGroup(h5_group, decimation_level_metadata=metadata)
   >>> channel = decimation.add_channel('Ex', fc_data=fc_array)


   .. py:method:: metadata()

      Overwrite get metadata to include channel information in the runs



   .. py:property:: channel_summary
      :type: pandas.DataFrame


      Get a summary of all channels in this decimation level.

      Returns a pandas DataFrame with detailed information about each Fourier
      Coefficient channel including time ranges, dimensions, and sampling rates.

      :returns: DataFrame with columns:

                - component : str
                    Channel component name (e.g., 'Ex', 'Hy')
                - start : datetime64[ns]
                    Start time of the channel data
                - end : datetime64[ns]
                    End time of the channel data
                - n_frequency : int64
                    Number of frequency bins
                - n_windows : int64
                    Number of time windows
                - sample_rate_decimation_level : float64
                    Decimation level sample rate (Hz)
                - sample_rate_window_step : float64
                    Sample rate of window stepping (Hz)
                - units : str
                    Physical units of the data
                - hdf5_reference : h5py.ref_dtype
                    HDF5 reference to the channel dataset
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> summary = decimation.channel_summary
      >>> print(summary[['component', 'n_frequency', 'n_windows']])


   .. py:method:: from_dataframe(df: pandas.DataFrame, channel_key: str, time_key: str = 'time', frequency_key: str = 'frequency') -> None

      Load Fourier Coefficient data from a pandas DataFrame.

      Assumes the channel_key column contains complex coefficient values
      organized with time and frequency dimensions.

      :param df: Input DataFrame containing the coefficient data.
      :type df: pd.DataFrame
      :param channel_key: Name of the column containing coefficient values.
      :type channel_key: str
      :param time_key: Name of the time coordinate column.
      :type time_key: str, default='time'
      :param frequency_key: Name of the frequency coordinate column.
      :type frequency_key: str, default='frequency'

      :raises TypeError: If df is not a pandas DataFrame.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> decimation.from_dataframe(df, channel_key='Ex', time_key='time')



   .. py:method:: from_xarray(data_array: xarray.Dataset | xarray.DataArray, sample_rate_decimation_level: float) -> None

      Load Fourier Coefficient data from an xarray DataArray or Dataset.

      Automatically extracts metadata (time, frequency, units) from the xarray
      object and creates appropriate FCChannelDataset instances for each
      variable or the single DataArray.

      :param data_array: Input xarray object with 'time' and 'frequency' coordinates and
                         dimensions ['time', 'frequency'] (or transposed variant).
      :type data_array: xr.DataArray or xr.Dataset
      :param sample_rate_decimation_level: Sample rate of the decimation level (Hz).
      :type sample_rate_decimation_level: float

      :raises TypeError: If data_array is not an xarray Dataset or DataArray.

      .. rubric:: Notes

      Automatically handles both (time, frequency) and (frequency, time)
      dimension ordering. Units are extracted from xarray attributes if available.

      .. rubric:: Examples

      >>> import xarray as xr
      >>> import numpy as np
      >>> decimation = FCDecimationGroup(h5_group)

      Create sample xarray data:

      >>> times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[s]')
      >>> freqs = np.linspace(0.01, 100, 256)
      >>> data_array = np.random.randn(len(times), len(freqs)) + \
      ...              1j * np.random.randn(len(times), len(freqs))
      >>> xr_data = xr.DataArray(
      ...     data_array,
      ...     dims=['time', 'frequency'],
      ...     coords={'time': times, 'frequency': freqs},
      ...     name='Ex'
      ... )

      Load into decimation group:

      >>> decimation.from_xarray(xr_data, sample_rate_decimation_level=0.5)



   .. py:method:: to_xarray(channels: Optional[list[str]] = None) -> xarray.Dataset

      Create an xarray Dataset from Fourier Coefficient channels.

      If no channels are specified, all channels in the decimation level
      are included. Each channel becomes a data variable in the resulting Dataset.

      :param channels: List of channel names to include. If None, all channels are used.
                       Default is None.
      :type channels: list[str], optional

      :returns: xarray Dataset with channels as data variables and 'time' and
                'frequency' as shared coordinates.
      :rtype: xr.Dataset

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> xr_data = decimation.to_xarray()
      >>> print(xr_data.data_vars)
      Data variables:
          Ex  (time, frequency) complex128
          Ey  (time, frequency) complex128

      Get specific channels:

      >>> subset = decimation.to_xarray(channels=['Ex', 'Ey'])



   .. py:method:: from_numpy_array(nd_array: numpy.ndarray, ch_name: str | list[str]) -> None

      Load Fourier Coefficient data from a numpy array.

      Assumes array shape is either (n_frequencies, n_windows) for a single
      channel or (n_channels, n_frequencies, n_windows) for multiple channels.

      :param nd_array: Input numpy array containing coefficient data.
      :type nd_array: np.ndarray
      :param ch_name: Channel name (for 2D array) or list of channel names
                      (for 3D array).
      :type ch_name: str or list[str]

      :raises TypeError: If nd_array is not a numpy ndarray.
      :raises ValueError: If array shape is not (n_frequencies, n_windows) or
          (n_channels, n_frequencies, n_windows).

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)

      Load single channel:

      >>> data_2d = np.random.randn(256, 100) + 1j * np.random.randn(256, 100)
      >>> decimation.from_numpy_array(data_2d, ch_name='Ex')

      Load multiple channels:

      >>> data_3d = np.random.randn(2, 256, 100) + 1j * np.random.randn(2, 256, 100)
      >>> decimation.from_numpy_array(data_3d, ch_name=['Ex', 'Ey'])



   .. py:method:: add_channel(fc_name: str, fc_data: Optional[numpy.ndarray] = None, fc_metadata: Optional[mt_metadata.processing.fourier_coefficients.FCChannel] = None, max_shape: tuple = (None, None), chunks: bool = True, dtype: type = complex, **kwargs) -> mth5.groups.FCChannelDataset

      Add a Fourier Coefficient channel to the decimation level.

      Creates a new FCChannelDataset for a single channel at a single
      decimation level. Input data can be provided as numpy array or created empty.

      :param fc_name: Name for the Fourier Coefficient channel (usually component name like 'Ex').
      :type fc_name: str
      :param fc_data: Input data with shape (n_frequencies, n_windows). Default is None (creates empty).
      :type fc_data: np.ndarray, optional
      :param fc_metadata: Metadata for the channel. Default is None.
      :type fc_metadata: fc.FCChannel, optional
      :param max_shape: Maximum shape for HDF5 dataset dimensions (expandable if None).
      :type max_shape: tuple, default=(None, None)
      :param chunks: Whether to use HDF5 chunking.
      :type chunks: bool, default=True
      :param dtype: Data type for the dataset.
      :type dtype: type, default=complex
      :param \*\*kwargs: Additional keyword arguments for HDF5 dataset creation.

      :returns: Newly created FCChannelDataset object.
      :rtype: FCChannelDataset

      :raises TypeError: If fc_data type is not supported.

      .. rubric:: Notes

      Data layout assumes (time, frequency) organization:

      - time index: window start times
      - frequency index: harmonic indices or float values
      - data: complex Fourier coefficients

      If a channel with the same name already exists, the existing channel
      is returned instead of creating a duplicate.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> metadata = fc.FCChannel(component='Ex')

      Create from numpy array:

      >>> fc_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
      >>> channel = decimation.add_channel('Ex', fc_data=fc_data, fc_metadata=metadata)

      Create empty channel (expandable):

      >>> channel = decimation.add_channel('Ex', fc_metadata=metadata)



   .. py:method:: get_channel(fc_name: str) -> mth5.groups.FCChannelDataset

      Retrieve a Fourier Coefficient channel by name.

      :param fc_name: Name of the Fourier Coefficient channel to retrieve.
      :type fc_name: str

      :returns: The requested Fourier Coefficient channel dataset.
      :rtype: FCChannelDataset

      :raises KeyError: If the channel does not exist in this decimation level.
      :raises MTH5Error: If unable to retrieve the channel from HDF5.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> channel = decimation.get_channel('Ex')
      >>> print(channel.shape)
      (100, 256)



   .. py:method:: remove_channel(fc_name: str) -> None

      Remove a Fourier Coefficient channel from the decimation level.

      Deletes the HDF5 dataset associated with the channel. Note that this
      removes the reference but does not reduce the HDF5 file size.

      :param fc_name: Name of the Fourier Coefficient channel to remove.
      :type fc_name: str

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Notes

      Deleting a channel does not reduce the HDF5 file size; it simply
      removes the reference to the data. To truly reduce file size, copy
      the desired data to a new file.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> decimation.remove_channel('Ex')



   .. py:method:: update_metadata() -> None

      Update decimation level metadata from all channels.

      Aggregates metadata from all FC channels in the decimation level
      including time period, sample rates, and window step information.
      Updates the internal metadata object and writes to HDF5.

      .. rubric:: Notes

      Collects the following information from channels:

      - Time period start/end from channel data
      - Sample rate decimation level
      - Sample rate window step

      Should be called after adding or modifying channels to keep
      metadata synchronized.

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> decimation.add_channel('Ex', fc_data=data_ex)
      >>> decimation.add_channel('Ey', fc_data=data_ey)
      >>> decimation.update_metadata()



   .. py:method:: add_feature(feature_name: str, feature_data: Optional[numpy.ndarray] = None, feature_metadata: Optional[dict] = None, max_shape: tuple = (None, None, None), chunks: bool = True, **kwargs) -> None

      Add a feature dataset to the decimation level.

      Creates a new dataset for auxiliary features or derived quantities
      related to Fourier Coefficients (e.g., SNR, coherency, power, etc.).

      :param feature_name: Name for the feature dataset.
      :type feature_name: str
      :param feature_data: Input data for the feature. Default is None (creates empty).
      :type feature_data: np.ndarray, optional
      :param feature_metadata: Metadata dictionary for the feature. Default is None.
      :type feature_metadata: dict, optional
      :param max_shape: Maximum shape for HDF5 dataset dimensions (expandable if None).
      :type max_shape: tuple, default=(None, None, None)
      :param chunks: Whether to use HDF5 chunking.
      :type chunks: bool, default=True
      :param \*\*kwargs: Additional keyword arguments for HDF5 dataset creation.

      .. rubric:: Notes

      Feature types may include:

      - Power: Total power in Fourier coefficients
      - SNR: Signal-to-noise ratio
      - Coherency: Cross-component coherence
      - Weights: Channel-specific weights
      - Flags: Data quality or processing flags

      .. rubric:: Examples

      >>> decimation = FCDecimationGroup(h5_group)
      >>> snr_data = np.random.randn(100, 256)
      >>> decimation.add_feature('snr', feature_data=snr_data)

      Or create empty feature for later population:

      >>> decimation.add_feature('power_Ex')



.. py:class:: TransferFunctionsGroup(group: Any, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for transfer functions under a station.

   Each child group is a single transfer function estimation managed by
   :class:`TransferFunctionGroup`.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> m5 = mth5.MTH5()
   >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
   >>> station = m5.stations_group.add_station("mt01")
   >>> tf_group = station.transfer_functions_group
   >>> tf_group.groups_list
   []


   .. py:method:: tf_summary(as_dataframe: bool = True) -> pandas.DataFrame | numpy.ndarray

      Summarize transfer functions stored for the station.

      :param as_dataframe: If ``True`` return a pandas DataFrame, otherwise a NumPy structured array.
      :type as_dataframe: bool, default True

      :returns: Summary rows including station reference, location, and TF metadata.
      :rtype: pandas.DataFrame or numpy.ndarray

      .. rubric:: Examples

      >>> summary = tf_group.tf_summary()
      >>> summary.columns[:4].tolist()  # doctest: +SKIP
      ['station_hdf5_reference', 'station', 'latitude', 'longitude']



   .. py:method:: add_transfer_function(name: str, tf_object: mt_metadata.transfer_functions.core.TF | None = None) -> TransferFunctionGroup

      Add a transfer function group under this station.

      :param name: Transfer function identifier.
      :type name: str
      :param tf_object: Transfer function instance to seed metadata and datasets.
      :type tf_object: TF, optional

      :returns: Wrapper for the created or existing transfer function.
      :rtype: TransferFunctionGroup

      .. rubric:: Examples

      >>> tf_group = station.transfer_functions_group
      >>> _ = tf_group.add_transfer_function("mt01_4096")



   .. py:method:: get_transfer_function(tf_id: str) -> TransferFunctionGroup

      Return an existing transfer function by id.

      :param tf_id: Name of the transfer function.
      :type tf_id: str

      :returns: Wrapper for the requested transfer function.
      :rtype: TransferFunctionGroup

      :raises MTH5Error: If the transfer function does not exist.

      .. rubric:: Examples

      >>> existing = station.transfer_functions_group.get_transfer_function("mt01_4096")
      >>> existing.name  # doctest: +SKIP
      'mt01_4096'



   .. py:method:: remove_transfer_function(tf_id: str) -> None

      Delete a transfer function reference from the station.

      :param tf_id: Transfer function name.
      :type tf_id: str

      .. rubric:: Notes

      HDF5 deletion removes the reference only; storage is not reclaimed.

      .. rubric:: Examples

      >>> tf_group.remove_transfer_function("mt01_4096")



   .. py:method:: get_tf_object(tf_id: str) -> mt_metadata.transfer_functions.core.TF

      Return a populated :class:`mt_metadata.transfer_functions.core.TF`.

      :param tf_id: Transfer function name to convert.
      :type tf_id: str

      :returns: Transfer function populated with metadata and estimates.
      :rtype: mt_metadata.transfer_functions.core.TF

      .. rubric:: Examples

      >>> tf_obj = tf_group.get_tf_object("mt01_4096")  # doctest: +SKIP



.. py:class:: TransferFunctionGroup(group: Any, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Wrapper for a single transfer function estimation.


   .. py:method:: has_estimate(estimate: str) -> bool

      Return ``True`` if an estimate exists and is populated.



   .. py:property:: period
      :type: numpy.ndarray | None


      Return period array stored in ``period`` dataset, if present.


   .. py:method:: add_statistical_estimate(estimate_name: str, estimate_data: numpy.ndarray | xarray.DataArray | None = None, estimate_metadata: mt_metadata.transfer_functions.tf.statistical_estimate.StatisticalEstimate | None = None, max_shape: tuple[int | None, int | None, int | None] = (None, None, None), chunks: bool = True, **kwargs: Any) -> mth5.groups.EstimateDataset

      Add a statistical estimate dataset.

      :param estimate_name: Dataset name.
      :type estimate_name: str
      :param estimate_data: Estimate values; if ``None`` a placeholder array is created.
      :type estimate_data: numpy.ndarray or xarray.DataArray, optional
      :param estimate_metadata: Metadata describing the estimate.
      :type estimate_metadata: StatisticalEstimate, optional
      :param max_shape: Maximum shape for resizable datasets.
      :type max_shape: tuple of int or None, default (None, None, None)
      :param chunks: Chunking flag forwarded to HDF5 dataset creation.
      :type chunks: bool, default True

      :returns: Wrapper combining dataset and metadata.
      :rtype: EstimateDataset

      :raises TypeError: If ``estimate_data`` is not array-like.

      .. rubric:: Examples

      >>> est = tf_group.add_statistical_estimate("transfer_function")
      >>> isinstance(est, EstimateDataset)
      True



   .. py:method:: get_estimate(estimate_name: str) -> mth5.groups.EstimateDataset

      Return a statistical estimate dataset by name.



   .. py:method:: remove_estimate(estimate_name: str) -> None

      Remove a statistical estimate dataset reference.



   .. py:method:: to_tf_object() -> mt_metadata.transfer_functions.core.TF

      Convert this group into a populated :class:`TF` object.

      :returns: TF instance with survey, station, runs, channels, period, and
                estimate datasets applied.
      :rtype: mt_metadata.transfer_functions.core.TF

      :raises ValueError: If no period dataset is present.

      .. rubric:: Examples

      >>> tf_obj = tf_group.to_tf_object()  # doctest: +SKIP



   .. py:method:: from_tf_object(tf_obj: mt_metadata.transfer_functions.core.TF, update_metadata: bool = True) -> None

      Populate datasets from a :class:`TF` object.

      :param tf_obj: Transfer function object containing estimates and metadata.
      :type tf_obj: TF
      :param update_metadata: If ``True`` write transfer function metadata to HDF5.
      :type update_metadata: bool, default True

      :raises ValueError: If ``tf_obj`` is not a ``TF`` instance.

      .. rubric:: Examples

      >>> tf_group.from_tf_object(tf_obj)  # doctest: +SKIP



.. py:class:: RunGroup(group: h5py.Group, run_metadata: Optional[mt_metadata.timeseries.Run] = None, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for a single MT measurement run with multiple channels.

   Manages time series data and metadata for one measurement run within a station.
   A run can contain multiple channels of electric, magnetic, and auxiliary data.
   This class provides methods to add, retrieve, and manage individual channels,
   along with convenient access to station and survey metadata.

   The run group is located at ``/Survey/Stations/{station_name}/{run_name}`` in
   the HDF5 file hierarchy.

   .. attribute:: metadata

      Run metadata including sample rate, time period, and channel information.

      :type: mt_metadata.timeseries.Run

   .. attribute:: channel_summary

      Summary table of all channels in the run.

      :type: pd.DataFrame

   .. attribute:: groups_list

      List of channel names in the run.

      :type: list[str]

   :param group: HDF5 group for the run, should have path like
                 ``/Survey/Stations/{station_name}/{run_name}``
   :type group: h5py.Group
   :param run_metadata: Metadata container for the run. Default is None.
   :type run_metadata: mt_metadata.timeseries.Run, optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.
   :type \*\*kwargs: Any

   .. rubric:: Notes

   Key behaviors:

   - Channels can be of type: electric, magnetic, or auxiliary
   - All metadata updates should use the metadata object for validation
   - Call write_metadata() after modifying metadata to persist changes
   - Channel metadata is cached for performance during repeated access
   - Deleting a channel removes the reference but doesn't reduce file size

   .. rubric:: Examples

   Access run from an open MTH5 file:

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5()
   >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
   >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')

   Check available channels:

   >>> run.groups_list
   ['Ex', 'Ey', 'Hx', 'Hy']

   Access HDF5 group directly:

   >>> run.hdf5_group.ref
   <HDF5 Group Reference>

   Update metadata and persist to file:

   >>> run.metadata.sample_rate = 512.0
   >>> run.write_metadata()

   Add a channel:

   >>> import numpy as np
   >>> data = np.random.rand(4096)
   >>> ex = run.add_channel('Ex', 'electric', data=data)

   This class provides methods to add and get channels.  A summary table of
   all existing channels in the run is also provided as a convenience look up
   table to make searching easier.

   :param group: HDF5 group for a station, should have a path
                 ``/Survey/Stations/station_name/run_name``
   :type group: :class:`h5py.Group`
   :param station_metadata: metadata container, defaults to None
   :type station_metadata: :class:`mth5.metadata.Station`, optional

   :Access RunGroup from an open MTH5 file:

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5()
   >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
   >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')

   :Check what channels exist:

   >>> station.groups_list
   ['Ex', 'Ey', 'Hx', 'Hy']

   To access the hdf5 group directly use `RunGroup.hdf5_group`

   >>> station.hdf5_group.ref
   <HDF5 Group Reference>

   .. note:: All attributes should be input into the metadata object, that
            way all input will be validated against the metadata standards.
            If you change attributes in metadata object, you should run the
            `SurveyGroup.write_metadata()` method.  This is a temporary
            solution, working on an automatic updater if metadata is changed.

   >>> run.metadata.existing_attribute = 'update_existing_attribute'
   >>> run.write_metadata()

   If you want to add a new attribute this should be done using the
   `metadata.add_base_attribute` method.

   >>> station.metadata.add_base_attribute('new_attribute',
   >>> ...                                 'new_attribute_value',
   >>> ...                                 {'type':str,
   >>> ...                                  'required':True,
   >>> ...                                  'style':'free form',
   >>> ...                                  'description': 'new attribute desc.',
   >>> ...                                  'units':None,
   >>> ...                                  'options':[],
   >>> ...                                  'alias':[],
   >>> ...                                  'example':'new attribute

   :Add a channel:

   >>> new_channel = run.add_channel('Ex', 'electric',
   >>> ...                            data=numpy.random.rand(4096))
   >>> new_run
   /Survey/Stations/MT001/MT001a:
   =======================================
       --> Dataset: summary
       ......................
       --> Dataset: Ex
       ......................
       --> Dataset: Ey
       ......................
       --> Dataset: Hx
       ......................
       --> Dataset: Hy
       ......................

   :Add a channel with metadata:

   >>> from mth5.metadata import Electric
   >>> ex_metadata = Electric()
   >>> ex_metadata.time_period.start = '2020-01-01T12:30:00'
   >>> ex_metadata.time_period.end = '2020-01-03T16:30:00'
   >>> new_ex = run.add_channel('Ex', 'electric',
   >>> ...                       channel_metadata=ex_metadata)
   >>> # to look at the metadata
   >>> new_ex.metadata
   {
        "electric": {
           "ac.end": 1.2,
           "ac.start": 2.3,
           ...
           }
   }


   .. seealso:: `mth5.metadata` for details on how to add metadata from
                various files and python objects.

   :Remove a channel:

   >>> run.remove_channel('Ex')
   >>> station
   /Survey/Stations/MT001/MT001a:
   =======================================
       --> Dataset: summary
       ......................
       --> Dataset: Ey
       ......................
       --> Dataset: Hx
       ......................
       --> Dataset: Hy
       ......................

   .. note:: Deleting a station is not as simple as del(station).  In HDF5
             this does not free up memory, it simply removes the reference
             to that station.  The common way to get around this is to
             copy what you want into a new file, or overwrite the station.

   :Get a channel:

   >>> existing_ex = stations.get_channel('Ex')
   >>> existing_ex
   Channel Electric:
   -------------------
       data type:        Ex
       data type:        electric
       data format:      float32
       data shape:       (4096,)
       start:            1980-01-01T00:00:00+00:00
       end:              1980-01-01T00:32:+08:00
       sample rate:      8


   :summary Table:

   A summary table is provided to make searching easier.  The table
   summarized all stations within a survey. To see what names are in the
   summary table:

   >>> run.summary_table.dtype.descr
   [('component', ('|S5', {'h5py_encoding': 'ascii'})),
    ('start', ('|S32', {'h5py_encoding': 'ascii'})),
    ('end', ('|S32', {'h5py_encoding': 'ascii'})),
    ('n_samples', '<i4'),
    ('measurement_type', ('|S12', {'h5py_encoding': 'ascii'})),
    ('units', ('|S25', {'h5py_encoding': 'ascii'})),
    ('hdf5_reference', ('|O', {'ref': h5py.h5r.Reference}))]


   .. note:: When a run is added an entry is added to the summary table,
             where the information is pulled from the metadata.

   >>> new_run.summary_table
   index | component | start | end | n_samples | measurement_type | units |
   hdf5_reference
   --------------------------------------------------------------------------
   -------------


   .. py:property:: station_metadata
      :type: mt_metadata.timeseries.Station


      Get station metadata with current run included.

      :returns: Station metadata object containing this run's information.
      :rtype: metadata.Station

      .. rubric:: Examples

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5("example.h5", mode='r')
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> station_meta = run.station_metadata
      >>> print(station_meta.id)
      MT001


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Get survey metadata with current station and run included.

      :returns: Survey metadata object containing the full hierarchy.
      :rtype: metadata.Survey

      .. rubric:: Examples

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5("example.h5", mode='r')
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> survey_meta = run.survey_metadata
      >>> print(survey_meta.id)
      CONUS_South


   .. py:method:: recache_channel_metadata() -> None

      Clear and rebuild the channel metadata cache from current HDF5 data.

      This method reads all channel metadata from HDF5 storage and updates
      the internal cache. Useful when channel metadata has been modified
      externally or needs to be synchronized.

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> run.recache_channel_metadata()
      >>> # Cache is now synchronized with HDF5 storage



   .. py:method:: metadata() -> mt_metadata.timeseries.Run

      Get run metadata including all channel information.

      This property dynamically reads and caches channel metadata from HDF5,
      ensuring the run metadata always reflects the current state of channels.

      :returns: Run metadata object with all channels included.
      :rtype: metadata.Run

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> run_meta = run.metadata
      >>> print(run_meta.channels_recorded_electric)
      ['ex', 'ey']
      >>> print(run_meta.sample_rate)
      256.0



   .. py:property:: channel_summary
      :type: pandas.DataFrame


      Get summary of all channels in the run as a DataFrame.

      :returns: DataFrame with columns: component, start, end, n_samples,
                sample_rate, measurement_type, units, hdf5_reference.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> summary = run.channel_summary
      >>> print(summary[['component', 'sample_rate', 'n_samples']])
        component  sample_rate  n_samples
      0        ex        256.0      65536
      1        ey        256.0      65536
      2        hx        256.0      65536
      3        hy        256.0      65536


   .. py:method:: write_metadata() -> None

      Write run metadata to HDF5 attributes.

      Converts metadata object to dictionary and writes all attributes
      to the HDF5 group.

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> run.metadata.sample_rate = 512.0
      >>> run.write_metadata()
      >>> # Metadata is now persisted to HDF5 file



   .. py:method:: add_channel(channel_name, channel_type, data, channel_dtype='int32', shape=None, max_shape=(None, ), chunks=True, channel_metadata=None, **kwargs)

      Add a channel to the run.

      :param channel_name: Name of the channel (e.g., 'ex', 'ey', 'hx', 'hy', 'hz').
      :type channel_name: str
      :param channel_type: Type of channel: 'electric', 'magnetic', or 'auxiliary'.
      :type channel_type: str
      :param data: Time series data for the channel. If None, an empty resizable
                   dataset will be created.
      :type data: numpy.ndarray or None
      :param channel_dtype: Data type for the channel if data is None, by default "int32".
      :type channel_dtype: str, optional
      :param shape: Initial shape of the dataset. If None and data is None, shape
                    is estimated from metadata or set to (1,), by default None.
      :type shape: tuple of int, optional
      :param max_shape: Maximum shape the dataset can be resized to. Use None for
                        unlimited growth in that dimension, by default (None,).
      :type max_shape: tuple of int or None, optional
      :param chunks: Enable chunked storage. If True, uses automatic chunking.
                     If int, uses that chunk size, by default True.
      :type chunks: bool or int, optional
      :param channel_metadata: Metadata object for the channel, by default None.
      :type channel_metadata: mt_metadata.timeseries.Electric, Magnetic, or Auxiliary, optional
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: dict

      :returns: The created channel dataset object.
      :rtype: ElectricDataset or MagneticDataset or AuxiliaryDataset

      :raises MTH5Error: If channel_type is not one of: electric, magnetic, auxiliary.

      .. rubric:: Examples

      Add a channel with data:

      >>> import numpy as np
      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5("example.h5", mode='a')
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> data = np.random.rand(4096)
      >>> ex = run.add_channel('ex', 'electric', data)
      >>> print(ex.metadata.component)
      ex

      Add a channel with metadata:

      >>> from mt_metadata.timeseries import Electric
      >>> ex_meta = Electric()
      >>> ex_meta.time_period.start = '2020-01-01T12:30:00'
      >>> ex_meta.sample_rate = 256.0
      >>> ex = run.add_channel('ex', 'electric', None,
      ...                      channel_metadata=ex_meta)
      >>> print(ex.metadata.sample_rate)
      256.0

      Add a channel with custom shape:

      >>> ex = run.add_channel('ex', 'electric', None,
      ...                      shape=(8192,), channel_dtype='float32')
      >>> print(ex.hdf5_dataset.shape)
      (8192,)



   .. py:method:: get_channel(channel_name: str) -> mth5.groups.ElectricDataset | mth5.groups.MagneticDataset | mth5.groups.AuxiliaryDataset | mth5.groups.ChannelDataset

      Get a channel from an existing name.

      Returns the appropriate channel dataset container based on the
      channel type (electric, magnetic, or auxiliary).

      :param channel_name: Name of the channel to retrieve (e.g., 'ex', 'ey', 'hx').
      :type channel_name: str

      :returns: Channel dataset object containing the channel data and metadata.
      :rtype: ElectricDataset or MagneticDataset or AuxiliaryDataset or ChannelDataset

      :raises MTH5Error: If the channel does not exist in the run.

      .. rubric:: Examples

      Attempting to get a non-existent channel:

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5("example.h5", mode='r')
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> ex = run.get_channel('ex')
      MTH5Error: ex does not exist, check groups_list for existing names

      Check available channels first:

      >>> run.groups_list
      ['ey', 'hx', 'hz']

      Get an existing channel:

      >>> ey = run.get_channel('ey')
      >>> print(ey)
      Channel Electric:
      -------------------
              component:        ey
              data type:        electric
              data format:      float32
              data shape:       (4096,)
              start:            1980-01-01T00:00:00+00:00
              end:              1980-01-01T00:00:01+00:00
              sample rate:      4096



   .. py:method:: remove_channel(channel_name: str) -> None

      Remove a channel from the run.

      Deleting a channel is not as simple as del(channel). In HDF5,
      this does not free up memory; it simply removes the reference
      to that channel. The common way to get around this is to
      copy what you want into a new file, or overwrite the channel.

      :param channel_name: Name of the existing channel to remove.
      :type channel_name: str

      .. rubric:: Notes

      Deleting a channel does not reduce the HDF5 file size. It simply
      removes the reference. If file size reduction is your goal, copy
      what you want into another file.

      Todo: Need to remove summary table entry as well.

      .. rubric:: Examples

      >>> from mth5 import mth5
      >>> mth5_obj = mth5.MTH5()
      >>> mth5_obj.open_mth5(r"/test.mth5", mode='a')
      >>> run = mth5_obj.stations_group.get_station('MT001').get_run('MT001a')
      >>> run.remove_channel('ex')



   .. py:method:: has_data() -> bool

      Check if the run contains any non-empty, non-zero data.

      Verifies that all channels in the run have valid data (non-zero and
      non-empty arrays). Returns False if any channel lacks data.

      :returns: True if all channels have data, False if any channel is empty
                or all zeros.
      :rtype: bool

      .. rubric:: Notes

      A channel is considered to have data if its has_data() method
      returns True, meaning it contains non-zero values.

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> if run.has_data():
      ...     print("Run contains valid data")
      ...     runts = run.to_runts()



   .. py:method:: to_runts(start: Optional[str] = None, end: Optional[str] = None, n_samples: Optional[int] = None) -> mth5.timeseries.RunTS

      Convert run to a RunTS timeseries object.

      Combines all channels in the run into a RunTS object which handles
      multi-channel time series data with associated metadata.

      :param start: Start time for time slice in ISO format (e.g., '2023-01-01T12:00:00').
                    If None, uses entire channel data. Default is None.
      :type start: str, optional
      :param end: End time for time slice in ISO format. Only used if start is specified.
                  Default is None.
      :type end: str, optional
      :param n_samples: Number of samples to extract from start. If both end and n_samples
                        are specified, end takes precedence. Default is None.
      :type n_samples: int, optional

      :returns: RunTS object containing all channels with full run and station metadata.
      :rtype: RunTS

      .. rubric:: Notes

      - Includes run, station, and survey metadata in the output
      - Skips the 'summary' group which is not a channel
      - If start is specified, performs time slicing; otherwise returns full data

      .. rubric:: Examples

      Convert entire run to RunTS:

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> runts = run.to_runts()
      >>> print(runts.channels)
      ['ex', 'ey', 'hx', 'hy']

      Time slice the run:

      >>> runts = run.to_runts(start='2023-01-01T12:00:00',
      ...                       end='2023-01-01T13:00:00')
      >>> print(runts.ex.ts.shape)
      (1024,)



   .. py:method:: from_runts(run_ts_obj: mth5.timeseries.RunTS, **kwargs: Any) -> list[mth5.groups.ElectricDataset | mth5.groups.MagneticDataset | mth5.groups.AuxiliaryDataset]

      Create channel datasets from a RunTS timeseries object.

      Converts a RunTS object with multiple channels and metadata into
      HDF5 channel datasets and updates run metadata accordingly.

      :param run_ts_obj: RunTS object containing multiple channels and metadata.
      :type run_ts_obj: RunTS
      :param \*\*kwargs: Additional keyword arguments.
      :type \*\*kwargs: Any

      :returns: List of created channel dataset objects.
      :rtype: list[ElectricDataset | MagneticDataset | AuxiliaryDataset]

      :raises MTH5Error: If input is not a RunTS object.

      .. rubric:: Notes

      - Updates run metadata from input object
      - Validates station and run IDs match current context
      - Creates appropriate channel type based on channel metadata
      - Automatically registers recorded channels in run metadata

      .. rubric:: Examples

      >>> from mth5.timeseries import RunTS
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> runts = RunTS.from_file("timeseries_data.txt")
      >>> channels = run.from_runts(runts)
      >>> print(f"Created {len(channels)} channels")
      Created 4 channels



   .. py:method:: from_channel_ts(channel_ts_obj: mth5.timeseries.ChannelTS) -> mth5.groups.ElectricDataset | mth5.groups.MagneticDataset | mth5.groups.AuxiliaryDataset

      Create a channel dataset from a ChannelTS timeseries object.

      Converts a single ChannelTS object with time series data and metadata
      into an HDF5 channel dataset. Handles filter registration and updates
      run metadata with channel information.

      :param channel_ts_obj: ChannelTS object containing time series data and metadata.
      :type channel_ts_obj: ChannelTS

      :returns: Created channel dataset object.
      :rtype: ElectricDataset | MagneticDataset | AuxiliaryDataset

      :raises MTH5Error: If input is not a ChannelTS object.

      .. rubric:: Notes

      - Registers filters from channel response if present
      - Validates and corrects station/run ID mismatches
      - Updates run metadata recorded channel lists
      - Automatically determines channel type from metadata

      .. rubric:: Examples

      >>> from mth5.timeseries import ChannelTS
      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> channel = ChannelTS.from_file("ex_timeseries.txt")
      >>> ex = run.from_channel_ts(channel)
      >>> print(ex.metadata.component)
      ex



   .. py:method:: update_run_metadata() -> None

      Update metadata and table entries (Deprecated).
      .. deprecated::
          Use update_metadata() instead.
      :raises DeprecationWarning: Always raised to indicate this method should not be used.



   .. py:method:: update_metadata() -> None

      Update run metadata from all channels and persist to HDF5.

      Aggregates metadata from all channels including time period and
      sample rate, then writes updated metadata to HDF5 attributes.

      :raises Exception: May raise exceptions if no channels exist (logs warning).

      .. rubric:: Notes

      Updates:

      - Time period start from minimum of all channels
      - Time period end from maximum of all channels
      - Sample rate from first channel (assumes uniform across channels)

      Should be called after adding or removing channels to maintain
      consistency between channel and run metadata.

      .. rubric:: Examples

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> run.add_channel('ex', 'electric', data=ex_data)
      >>> run.add_channel('ey', 'electric', data=ey_data)
      >>> run.update_metadata()  # Updates time period and sample rate



   .. py:method:: plot(start: Optional[str] = None, end: Optional[str] = None, n_samples: Optional[int] = None) -> Any

      Create a matplotlib plot of all channels in the run.

      Generates a multi-panel plot showing all channels in the run using
      the RunTS plotting functionality.

      :param start: Start time for time slice in ISO format. If None, plots entire
                    channel data. Default is None.
      :type start: str, optional
      :param end: End time for time slice in ISO format. Only used if start is
                  specified. Default is None.
      :type end: str, optional
      :param n_samples: Number of samples to extract from start. If both end and n_samples
                        are specified, end takes precedence. Default is None.
      :type n_samples: int, optional

      :returns: Matplotlib figure or axes object (depends on RunTS.plot() implementation).
      :rtype: Any

      .. rubric:: Notes

      - Creates separate subplots for each channel type (electric, magnetic, auxiliary)
      - Time slice parameters work the same as to_runts()
      - Requires matplotlib to be installed

      .. rubric:: Examples

      Plot entire run:

      >>> run = mth5_obj.get_run("MT001", "MT001a")
      >>> fig = run.plot()
      >>> fig.show()

      Plot time slice:

      >>> fig = run.plot(start='2023-01-01T12:00:00',
      ...                end='2023-01-01T13:00:00')



.. py:class:: FeatureChannelDataset(dataset: h5py.Dataset, dataset_metadata: Optional[mt_metadata.features.FeatureDecimationChannel] = None, **kwargs)

   Container for multi-dimensional Fourier Coefficients organized by time and frequency.

   This class manages Fourier Coefficient data with frequency band organization,
   similar to FCDataset but with enhanced band tracking capabilities. The data array
   is organized with the following assumptions:

   1. Data are grouped into frequency bands
   2. Data are uniformly sampled in time (uniform FFT moving window step size)

   The dataset tracks temporal evolution of frequency content across multiple windows,
   making it suitable for time-frequency analysis of geophysical signals.

   :param dataset: HDF5 dataset containing the Fourier coefficient data.
   :type dataset: h5py.Dataset
   :param dataset_metadata: Metadata for the dataset. See :class:`mt_metadata.features.FeatureDecimationChannel`.
                            If provided, must be of the same type as the internal metadata class.
                            Default is None.
   :type dataset_metadata: FeatureDecimationChannel, optional
   :param \*\*kwargs: Additional keyword arguments for future extensibility.

   .. attribute:: hdf5_dataset

      Reference to the HDF5 dataset.

      :type: h5py.Dataset

   .. attribute:: metadata

      Metadata container with the following attributes:

      - name : str
          Dataset name
      - time_period.start : datetime
          Start time of the data acquisition
      - time_period.end : datetime
          End time of the data acquisition
      - sample_rate_window_step : float
          Sample rate of the time window stepping (Hz)
      - frequency_min : float
          Minimum frequency in the band (Hz)
      - frequency_max : float
          Maximum frequency in the band (Hz)
      - units : str
          Physical units of the coefficient data
      - component : str
          Component identifier (e.g., 'Ex', 'Hy')
      - sample_rate_decimation_level : int
          Decimation level applied to acquire this data

      :type: FeatureDecimationChannel

   :raises MTH5Error: If dataset_metadata type does not match the expected FeatureDecimationChannel type.

   .. rubric:: Examples

   >>> import h5py
   >>> from mt_metadata.features import FeatureDecimationChannel
   >>> from mth5.groups.feature_dataset import FeatureChannelDataset

   Create a feature dataset from an HDF5 group:

   >>> with h5py.File('data.h5', 'r') as f:
   ...     h5_dataset = f['feature_group']['Ex']
   ...     feature = FeatureChannelDataset(h5_dataset)
   ...     print(f"Time windows: {feature.n_windows}")
   ...     print(f"Frequencies: {feature.n_frequencies}")

   Access time and frequency arrays:

   >>> time_array = feature.time
   >>> freq_array = feature.frequency
   >>> data_array = feature.to_numpy()


   .. py:attribute:: logger


   .. py:attribute:: metadata


   .. py:method:: read_metadata() -> None

      Read metadata from the HDF5 file into the metadata container.

      This method loads all attributes from the HDF5 dataset into the
      metadata container, enabling validation and type checking.

      .. rubric:: Examples

      >>> feature.read_metadata()
      >>> print(feature.metadata.component)
      'Ex'



   .. py:method:: write_metadata() -> None

      Write metadata from the metadata container to the HDF5 attributes.

      This method serializes the metadata container and writes all metadata
      as attributes to the HDF5 dataset. Raises exceptions are caught for
      read-only files.

      .. rubric:: Examples

      >>> feature.metadata.component = 'Ey'
      >>> feature.write_metadata()



   .. py:property:: n_windows
      :type: int


      Get the number of time windows in the dataset.

      :returns: Number of time windows (first dimension of the dataset).
      :rtype: int


   .. py:property:: time
      :type: numpy.ndarray


      Get the time array for each window.

      Returns an array of datetime64 values representing the start time
      of each time window. The time spacing is determined by the sample
      rate of the window stepping.

      :returns: Array of datetime64 values with shape (n_windows,) representing
                the start time of each window.
      :rtype: np.ndarray

      .. rubric:: Examples

      >>> time_array = feature.time
      >>> print(time_array.shape)
      (100,)
      >>> print(time_array[0])
      numpy.datetime64('2023-01-01T00:00:00')


   .. py:property:: n_frequencies
      :type: int


      Get the number of frequency bins in the dataset.

      :returns: Number of frequency bins (second dimension of the dataset).
      :rtype: int


   .. py:property:: frequency
      :type: numpy.ndarray


      Get the frequency array for the dataset.

      Returns a linearly-spaced frequency array from frequency_min to
      frequency_max with n_frequencies points.

      :returns: Array of float64 frequencies in Hz with shape (n_frequencies,).
      :rtype: np.ndarray

      .. rubric:: Examples

      >>> freq_array = feature.frequency
      >>> print(freq_array.shape)
      (256,)
      >>> print(f"Frequency range: {freq_array[0]:.2f} - {freq_array[-1]:.2f} Hz")
      Frequency range: 0.01 - 100.00 Hz


   .. py:method:: replace_dataset(new_data_array: numpy.ndarray) -> None

      Replace the entire HDF5 dataset with new data.

      This method resizes the HDF5 dataset as needed and replaces all data.
      The input array must have the same dtype as the existing dataset.

      :param new_data_array: New data array to replace the existing dataset. Will be converted
                             to numpy array if necessary.
      :type new_data_array: np.ndarray

      :raises TypeError: If input cannot be converted to a numpy array or has incompatible shape.

      .. rubric:: Examples

      >>> import numpy as np
      >>> new_data = np.random.randn(100, 256)
      >>> feature.replace_dataset(new_data)



   .. py:method:: to_xarray() -> xarray.DataArray

      Convert the feature dataset to an xarray DataArray.

      Returns an xarray DataArray with proper time and frequency coordinates,
      metadata attributes, and component naming. The entire dataset is loaded
      into memory.

      :returns: DataArray with dimensions ['time', 'frequency'] and coordinates
                matching the dataset's time and frequency arrays.
      :rtype: xr.DataArray

      .. rubric:: Notes

      Metadata stored in xarray attributes will not be validated if modified.
      The full dataset is loaded into memory; use with caution for large datasets.

      .. rubric:: Examples

      >>> xr_data = feature.to_xarray()
      >>> print(xr_data.dims)
      ('time', 'frequency')
      >>> print(xr_data.name)
      'Ex'
      >>> subset = xr_data.sel(time=slice('2023-01-01', '2023-01-02'))



   .. py:method:: to_numpy() -> numpy.ndarray

      Convert the feature dataset to a numpy array.

      Returns the dataset as a numpy array by loading it from the HDF5 file
      into memory. The array shape is (n_windows, n_frequencies).

      :returns: Numpy array containing all feature data with shape
                (n_windows, n_frequencies).
      :rtype: np.ndarray

      .. rubric:: Examples

      >>> data = feature.to_numpy()
      >>> print(data.shape)
      (100, 256)
      >>> print(data.dtype)
      complex128
      >>> mean_amplitude = np.abs(data).mean()



   .. py:method:: from_numpy(new_estimate: numpy.ndarray) -> None

      Load data from a numpy array into the HDF5 dataset.

      This method updates the HDF5 dataset with new data from a numpy array.
      The input array must match the dataset's dtype. The HDF5 dataset will
      be resized if necessary to accommodate the new data.

      :param new_estimate: Numpy array to write to the HDF5 dataset. Must have compatible
                           dtype with the existing dataset.
      :type new_estimate: np.ndarray

      :raises TypeError: If input array dtype does not match the HDF5 dataset dtype or
          if input cannot be converted to numpy array.

      .. rubric:: Notes

      The variable 'data' is a builtin in numpy and cannot be used as a parameter name.

      .. rubric:: Examples

      >>> import numpy as np
      >>> new_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
      >>> feature.from_numpy(new_data)
      >>> loaded_data = feature.to_numpy()
      >>> assert loaded_data.shape == new_data.shape



   .. py:method:: from_xarray(data: xarray.DataArray, sample_rate_decimation_level: int) -> None

      Load data and metadata from an xarray DataArray.

      This method updates both the HDF5 dataset and metadata from an xarray
      DataArray. It extracts time coordinates, frequency range, and component
      information from the DataArray and its attributes.

      :param data: Input xarray DataArray with 'time' and 'frequency' coordinates.
                   Expected dimensions are ['time', 'frequency'].
      :type data: xr.DataArray
      :param sample_rate_decimation_level: Decimation level applied to the original data to produce this
                                           feature dataset (integer ≥ 1).
      :type sample_rate_decimation_level: int

      .. rubric:: Notes

      Metadata stored in xarray attributes will be extracted and written to
      the HDF5 file. The full dataset is loaded into memory during this process.

      .. rubric:: Examples

      >>> import xarray as xr
      >>> import numpy as np

      Create sample xarray data:

      >>> times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[s]')
      >>> freqs = np.linspace(0.01, 100, 256)
      >>> data_array = np.random.randn(len(times), len(freqs)) + \
      ...              1j * np.random.randn(len(times), len(freqs))
      >>> xr_data = xr.DataArray(
      ...     data_array,
      ...     dims=['time', 'frequency'],
      ...     coords={'time': times, 'frequency': freqs},
      ...     name='Ex',
      ...     attrs={'units': 'mV/km'}
      ... )

      Load into feature dataset:

      >>> feature.from_xarray(xr_data, sample_rate_decimation_level=2)
      >>> print(feature.metadata.component)
      'Ex'



.. py:class:: MasterFeaturesGroup(group: h5py.Group, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Master group container for features associated with Fourier Coefficients or time series.

   This class manages the top-level organization of geophysical feature data,
   organizing it into feature-specific groups. Features can include various
   frequency or time-domain analyses.

   Hierarchy
   ---------
   MasterFeatureGroup -> FeatureGroup -> FeatureRunGroup ->

   - FC: FeatureDecimationGroup -> FeatureChannelDataset
   - Time Series: FeatureChannelDataset

   :param group: HDF5 group object for this MasterFeaturesGroup.
   :type group: h5py.Group
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> import h5py
   >>> from mth5.groups.features import MasterFeaturesGroup
   >>> with h5py.File('data.h5', 'r') as f:
   ...     master = MasterFeaturesGroup(f['features'])
   ...     feature_list = master.groups_list


   .. py:method:: add_feature_group(feature_name: str, feature_metadata: Optional[mt_metadata.features.FeatureDecimationChannel] = None) -> FeatureGroup

      Add a feature group to the master features container.

      Creates a new FeatureGroup with the specified name and optional metadata.
      Feature groups organize all runs and decimation levels for a particular feature.

      :param feature_name: Name for the feature group. Will be validated and formatted.
      :type feature_name: str
      :param feature_metadata: Metadata describing the feature. Default is None.
      :type feature_metadata: FeatureDecimationChannel, optional

      :returns: Newly created feature group object.
      :rtype: FeatureGroup

      .. rubric:: Examples

      >>> master = MasterFeaturesGroup(h5_group)
      >>> feature = master.add_feature_group('coherency')
      >>> print(feature.name)
      'coherency'



   .. py:method:: get_feature_group(feature_name: str) -> FeatureGroup

      Retrieve a feature group by name.

      :param feature_name: Name of the feature group to retrieve.
      :type feature_name: str

      :returns: The requested feature group.
      :rtype: FeatureGroup

      :raises MTH5Error: If the feature group does not exist.

      .. rubric:: Examples

      >>> master = MasterFeaturesGroup(h5_group)
      >>> feature = master.get_feature_group('coherency')
      >>> print(feature.name)
      'coherency'



   .. py:method:: remove_feature_group(feature_name: str) -> None

      Remove a feature group from the master container.

      Deletes the specified feature group and its associated data from the
      HDF5 file. Note that this operation removes the reference but does not
      reduce the file size; copy desired data to a new file for size reduction.

      :param feature_name: Name of the feature group to remove.
      :type feature_name: str

      :raises MTH5Error: If the feature group does not exist.

      .. rubric:: Examples

      >>> master = MasterFeaturesGroup(h5_group)
      >>> master.remove_feature_group('coherency')



.. py:class:: FeatureGroup(group: h5py.Group, feature_metadata: Optional[object] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for a single feature set with all associated runs and decimation levels.

   This class manages feature-specific data including all processing runs and
   decimation levels. Features can include both Fourier Coefficient and time series data.

   Hierarchy
   ---------
   FeatureGroup -> FeatureRunGroup ->

   - FC: FeatureDecimationLevel -> FeatureChannelDataset
   - TS: FeatureChannelDataset

   :param group: HDF5 group object for this FeatureGroup.
   :type group: h5py.Group
   :param feature_metadata: Metadata specific to this feature. Should include description and parameters.
   :type feature_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Notes

   Feature metadata should be specific to the feature and include descriptions
   of the feature and any parameters used in its computation.

   .. rubric:: Examples

   >>> feature = FeatureGroup(h5_group, feature_metadata=metadata)
   >>> run_group = feature.add_feature_run_group('run_1', domain='fc')


   .. py:method:: add_feature_run_group(feature_name: str, feature_run_metadata: Optional[object] = None, domain: str = 'fc') -> object

      Add a feature run group for a single feature.

      Creates either a Fourier Coefficient run group or a time series run group
      based on the specified domain. The domain can be determined from the metadata
      or explicitly provided.

      :param feature_name: Name for the feature run group.
      :type feature_name: str
      :param feature_run_metadata: Metadata for the feature run. If provided, domain is extracted from
                                   metadata.domain attribute. Default is None.
      :type feature_run_metadata: optional
      :param domain: Domain type for the data. Must be one of:

                     - 'fc', 'frequency', 'fourier', 'fourier_domain': Fourier Coefficients
                     - 'ts', 'time', 'time series', 'time_series': Time series
      :type domain: str, default='fc'

      :returns: Newly created feature run group.
      :rtype: FeatureFCRunGroup or FeatureTSRunGroup

      :raises ValueError: If domain is not recognized.
      :raises AttributeError: If metadata does not have a domain attribute when metadata is provided.

      .. rubric:: Examples

      >>> feature = FeatureGroup(h5_group)
      >>> fc_run = feature.add_feature_run_group('processing_run_1', domain='fc')
      >>> ts_run = feature.add_feature_run_group('ts_analysis', domain='ts')



   .. py:method:: get_feature_run_group(feature_name: str, domain: str = 'frequency') -> object

      Retrieve a feature run group by name and domain type.

      :param feature_name: Name of the feature run group to retrieve.
      :type feature_name: str
      :param domain: Domain type. Must be one of:

                     - 'fc', 'frequency', 'fourier', 'fourier_domain': Fourier Coefficients
                     - 'ts', 'time', 'time series', 'time_series': Time series
      :type domain: str, default='frequency'

      :returns: The requested feature run group.
      :rtype: FeatureFCRunGroup or FeatureTSRunGroup

      :raises ValueError: If domain is not recognized.
      :raises MTH5Error: If the feature run group does not exist.

      .. rubric:: Examples

      >>> feature = FeatureGroup(h5_group)
      >>> fc_run = feature.get_feature_run_group('processing_run_1', domain='fc')



   .. py:method:: remove_feature_run_group(feature_name: str) -> None

      Remove a feature run group.

      Deletes the specified feature run group and all its associated data.
      Note that deletion removes the reference but does not reduce HDF5 file size.

      :param feature_name: Name of the feature run group to remove.
      :type feature_name: str

      :raises MTH5Error: If the feature run group does not exist.

      .. rubric:: Examples

      >>> feature = FeatureGroup(h5_group)
      >>> feature.remove_feature_run_group('processing_run_1')



.. py:class:: FeatureTSRunGroup(group: h5py.Group, feature_run_metadata: Optional[object] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for time series features from a processing or analysis run.

   This class wraps a RunGroup to manage time series data features while
   maintaining compatibility with the feature hierarchy structure.

   :param group: HDF5 group object for this FeatureTSRunGroup.
   :type group: h5py.Group
   :param feature_run_metadata: Metadata for the feature run (same type as timeseries.Run).
   :type feature_run_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Notes

   This class uses methods from RunGroup for channel management, which may
   have performance implications due to multiple RunGroup instantiations.

   .. rubric:: Examples

   >>> ts_run = FeatureTSRunGroup(h5_group, feature_run_metadata=metadata)
   >>> channel = ts_run.add_feature_channel('Ex', 'electric', data)


   .. py:method:: add_feature_channel(channel_name: str, channel_type: str, data: Optional[numpy.ndarray] = None, channel_dtype: str = 'int32', shape: Optional[tuple] = None, max_shape: tuple = (None, ), chunks: bool = True, channel_metadata: Optional[object] = None, **kwargs) -> object

      Add a time series channel to the feature run group.

      Creates a new channel for time series data with the specified properties
      and optional metadata. Channel metadata should be a timeseries.Channel object.

      :param channel_name: Name for the channel.
      :type channel_name: str
      :param channel_type: Type of channel (e.g., 'electric', 'magnetic').
      :type channel_type: str
      :param data: Initial data for the channel. Default is None.
      :type data: np.ndarray, optional
      :param channel_dtype: Data type for the channel.
      :type channel_dtype: str, default='int32'
      :param shape: Shape of the channel data. Default is None.
      :type shape: tuple, optional
      :param max_shape: Maximum shape for expandable dimensions.
      :type max_shape: tuple, default=(None,)
      :param chunks: Whether to use chunking for the dataset.
      :type chunks: bool, default=True
      :param channel_metadata: Metadata object (timeseries.Channel type). Default is None.
      :type channel_metadata: optional
      :param \*\*kwargs: Additional keyword arguments for dataset creation.

      :returns: Channel object from RunGroup.
      :rtype: object

      .. rubric:: Examples

      >>> ts_run = FeatureTSRunGroup(h5_group)
      >>> channel = ts_run.add_feature_channel(
      ...     'Ex', 'electric', data=np.arange(1000))



   .. py:method:: get_feature_channel(channel_name: str) -> object

      Retrieve a feature channel by name.

      :param channel_name: Name of the channel to retrieve.
      :type channel_name: str

      :returns: Channel object from RunGroup.
      :rtype: object

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Examples

      >>> ts_run = FeatureTSRunGroup(h5_group)
      >>> channel = ts_run.get_feature_channel('Ex')



   .. py:method:: remove_feature_channel(channel_name: str) -> None

      Remove a feature channel from the run group.

      :param channel_name: Name of the channel to remove.
      :type channel_name: str

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Examples

      >>> ts_run = FeatureTSRunGroup(h5_group)
      >>> ts_run.remove_feature_channel('Ex')



.. py:class:: FeatureFCRunGroup(group: h5py.Group, feature_run_metadata: Optional[mt_metadata.processing.fourier_coefficients.decimation.Decimation] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for Fourier Coefficient features from a processing run.

   This class manages Fourier Coefficient data organized by decimation levels,
   each containing multiple frequency channels with time-frequency data.

   Hierarchy
   ---------
   FeatureFCRunGroup -> FeatureDecimationGroup -> FeatureChannelDataset

   .. attribute:: metadata

      Metadata including:

      - list of decimation levels
      - start time (earliest)
      - end time (latest)
      - method (fft, wavelet, ...)
      - list of channels used
      - starting sample rate
      - bands used
      - type (TS or FC)

      :type: Decimation

   :param group: HDF5 group object for this FeatureFCRunGroup.
   :type group: h5py.Group
   :param feature_run_metadata: Decimation metadata for the feature run. Default is None.
   :type feature_run_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> fc_run = FeatureFCRunGroup(h5_group, feature_run_metadata=metadata)
   >>> decimation = fc_run.add_decimation_level('level_0', dec_metadata)


   .. py:method:: metadata() -> mt_metadata.processing.fourier_coefficients.decimation.Decimation

      Overwrite get metadata to include channel information in the runs



   .. py:property:: decimation_level_summary
      :type: pandas.DataFrame


      Get a summary of all decimation levels in the run.

      Returns a pandas DataFrame with information about each decimation level
      including decimation factor, time range, and HDF5 reference.

      :returns: DataFrame with columns:

                - name : str
                    Decimation level name
                - start : datetime64[ns]
                    Start time of the decimation level
                - end : datetime64[ns]
                    End time of the decimation level
                - hdf5_reference : h5py.ref_dtype
                    HDF5 reference to the decimation level group
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> summary = fc_run.decimation_level_summary
      >>> print(summary[['name', 'start', 'end']])


   .. py:method:: add_decimation_level(decimation_level_name: str, feature_decimation_level_metadata: Optional[object] = None) -> FeatureDecimationGroup

      Add a decimation level group to the feature run.

      :param decimation_level_name: Name for the decimation level.
      :type decimation_level_name: str
      :param feature_decimation_level_metadata: Metadata for the decimation level. Default is None.
      :type feature_decimation_level_metadata: optional

      :returns: Newly created decimation level group.
      :rtype: FeatureDecimationGroup

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> decimation = fc_run.add_decimation_level('level_0', dec_metadata)
      >>> print(decimation.name)
      'level_0'



   .. py:method:: get_decimation_level(decimation_level_name: str) -> FeatureDecimationGroup

      Retrieve a decimation level group by name.

      :param decimation_level_name: Name of the decimation level to retrieve.
      :type decimation_level_name: str

      :returns: The requested decimation level group.
      :rtype: FeatureDecimationGroup

      :raises MTH5Error: If the decimation level does not exist.

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> decimation = fc_run.get_decimation_level('level_0')



   .. py:method:: remove_decimation_level(decimation_level_name: str) -> None

      Remove a decimation level from the feature run.

      :param decimation_level_name: Name of the decimation level to remove.
      :type decimation_level_name: str

      :raises MTH5Error: If the decimation level does not exist.

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> fc_run.remove_decimation_level('level_0')



   .. py:method:: update_metadata() -> None

      Update metadata from all decimation levels.

      Scans all decimation levels and updates the run-level metadata with
      aggregated information including time ranges.

      .. rubric:: Examples

      >>> fc_run = FeatureFCRunGroup(h5_group)
      >>> fc_run.update_metadata()



.. py:class:: FeatureDecimationGroup(group: h5py.Group, decimation_level_metadata: Optional[object] = None, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Container for a single decimation level with multiple Fourier Coefficient channels.

   This class manages Fourier Coefficient data organized by frequency, time, and channel.
   Data is assumed to be uniformly sampled in both frequency and time domains.

   Hierarchy
   ---------
   FeatureDecimationGroup -> FeatureChannelDataset (multiple channels)

   Data Assumptions
   ----------------
   1. Data are uniformly sampled in frequency domain
   2. Data are uniformly sampled in time domain
   3. FFT moving window has uniform step size

   .. attribute:: start time

      Start time of the decimation level

      :type: datetime

   .. attribute:: end time

      End time of the decimation level

      :type: datetime

   .. attribute:: channels

      List of channel names in this decimation level

      :type: list

   .. attribute:: decimation_factor

      Factor by which data was decimated

      :type: int

   .. attribute:: decimation_level

      Level index in decimation hierarchy

      :type: int

   .. attribute:: decimation_sample_rate

      Sample rate after decimation (Hz)

      :type: float

   .. attribute:: method

      Method used (FFT, wavelet, etc.)

      :type: str

   .. attribute:: anti_alias_filter

      Anti-aliasing filter used

      :type: optional

   .. attribute:: prewhitening_type

      Type of prewhitening applied

      :type: optional

   .. attribute:: harmonics_kept

      Harmonic indices kept in the data

      :type: list or 'all'

   .. attribute:: window

      Window parameters (length, overlap, type, sample rate)

      :type: dict

   .. attribute:: bands

      Frequency bands in the data

      :type: list

   :param group: HDF5 group object for this FeatureDecimationGroup.
   :type group: h5py.Group
   :param decimation_level_metadata: Metadata for the decimation level. Default is None.
   :type decimation_level_metadata: optional
   :param \*\*kwargs: Additional keyword arguments passed to BaseGroup.

   .. rubric:: Examples

   >>> decimation = FeatureDecimationGroup(h5_group, metadata)
   >>> channel = decimation.add_channel('Ex', fc_data=fc_array, fc_metadata=ch_metadata)


   .. py:method:: metadata()

      Overwrite get metadata to include channel information in the runs



   .. py:property:: channel_summary
      :type: pandas.DataFrame


      Get a summary of all channels in this decimation level.

      Returns a pandas DataFrame with detailed information about each Fourier
      Coefficient channel including time ranges, dimensions, and sampling rates.

      :returns: DataFrame with columns:

                - name : str
                    Channel name
                - start : datetime64[ns]
                    Start time of the channel data
                - end : datetime64[ns]
                    End time of the channel data
                - n_frequency : int64
                    Number of frequency bins
                - n_windows : int64
                    Number of time windows
                - sample_rate_decimation_level : float64
                    Decimation level sample rate (Hz)
                - sample_rate_window_step : float64
                    Sample rate of window stepping (Hz)
                - units : str
                    Physical units of the data
                - hdf5_reference : h5py.ref_dtype
                    HDF5 reference to the channel dataset
      :rtype: pd.DataFrame

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> summary = decimation.channel_summary
      >>> print(summary[['name', 'n_frequency', 'n_windows']])


   .. py:method:: from_dataframe(df: pandas.DataFrame, channel_key: str, time_key: str = 'time', frequency_key: str = 'frequency') -> None

      Load Fourier Coefficient data from a pandas DataFrame.

      Assumes the channel_key column contains complex coefficient values
      organized with time and frequency dimensions.

      :param df: Input DataFrame containing the coefficient data.
      :type df: pd.DataFrame
      :param channel_key: Name of the column containing coefficient values.
      :type channel_key: str
      :param time_key: Name of the time coordinate column.
      :type time_key: str, default='time'
      :param frequency_key: Name of the frequency coordinate column.
      :type frequency_key: str, default='frequency'

      :raises TypeError: If df is not a pandas DataFrame.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> decimation.from_dataframe(df, channel_key='Ex', time_key='time')



   .. py:method:: from_xarray(data_array: xarray.DataArray | xarray.Dataset, sample_rate_decimation_level: float) -> None

      Load Fourier Coefficient data from an xarray DataArray or Dataset.

      Automatically extracts metadata (time, frequency, units) from the xarray
      object and creates appropriate FeatureChannelDataset instances for each
      variable or the single DataArray.

      :param data_array: Input xarray object with 'time' and 'frequency' coordinates and
                         dimensions ['time', 'frequency'] (or transposed variant).
      :type data_array: xr.DataArray or xr.Dataset
      :param sample_rate_decimation_level: Sample rate of the decimation level (Hz).
      :type sample_rate_decimation_level: float

      :raises TypeError: If data_array is not an xarray Dataset or DataArray.

      .. rubric:: Notes

      Automatically handles both (time, frequency) and (frequency, time) dimension ordering.
      Units are extracted from xarray attributes if available.

      .. rubric:: Examples

      >>> import xarray as xr
      >>> import numpy as np
      >>> decimation = FeatureDecimationGroup(h5_group)

      Create sample xarray data:

      >>> times = np.arange('2023-01-01', '2023-01-02', dtype='datetime64[s]')
      >>> freqs = np.linspace(0.01, 100, 256)
      >>> data_array = np.random.randn(len(times), len(freqs)) + \
      ...              1j * np.random.randn(len(times), len(freqs))
      >>> xr_data = xr.DataArray(
      ...     data_array,
      ...     dims=['time', 'frequency'],
      ...     coords={'time': times, 'frequency': freqs},
      ...     name='Ex',
      ...     attrs={'units': 'mV/km'}
      ... )

      Load into decimation group:

      >>> decimation.from_xarray(xr_data, sample_rate_decimation_level=0.5)



   .. py:method:: to_xarray(channels: Optional[list] = None) -> xarray.Dataset

      Create an xarray Dataset from Fourier Coefficient channels.

      If no channels are specified, all channels in the decimation level
      are included. Each channel becomes a data variable in the resulting Dataset.

      :param channels: List of channel names to include. If None, all channels are used.
                       Default is None.
      :type channels: list, optional

      :returns: xarray Dataset with channels as data variables and 'time' and
                'frequency' as shared coordinates.
      :rtype: xr.Dataset

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> xr_data = decimation.to_xarray()
      >>> print(xr_data.data_vars)
      Data variables:
          Ex  (time, frequency) complex128
          Ey  (time, frequency) complex128

      Get specific channels:

      >>> subset = decimation.to_xarray(channels=['Ex', 'Ey'])



   .. py:method:: from_numpy_array(nd_array: numpy.ndarray, ch_name: str | list) -> None

      Load Fourier Coefficient data from a numpy array.

      Assumes array shape is either (n_frequencies, n_windows) for a single
      channel or (n_channels, n_frequencies, n_windows) for multiple channels.

      :param nd_array: Input numpy array containing coefficient data.
      :type nd_array: np.ndarray
      :param ch_name: Channel name (for 2D array) or list of channel names
                      (for 3D array).
      :type ch_name: str or list

      :raises TypeError: If nd_array is not a numpy ndarray.
      :raises ValueError: If array shape is not (n_frequencies, n_windows) or
          (n_channels, n_frequencies, n_windows).

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)

      Load single channel:

      >>> data_2d = np.random.randn(256, 100) + 1j * np.random.randn(256, 100)
      >>> decimation.from_numpy_array(data_2d, ch_name='Ex')

      Load multiple channels:

      >>> data_3d = np.random.randn(2, 256, 100) + 1j * np.random.randn(2, 256, 100)
      >>> decimation.from_numpy_array(data_3d, ch_name=['Ex', 'Ey'])



   .. py:method:: add_channel(fc_name: str, fc_data: Optional[numpy.ndarray | xarray.DataArray | xarray.Dataset | pandas.DataFrame] = None, fc_metadata: Optional[mt_metadata.features.FeatureDecimationChannel] = None, max_shape: tuple = (None, None), chunks: bool = True, dtype: type = complex, **kwargs) -> mth5.groups.FeatureChannelDataset

      Add a Fourier Coefficient channel to the decimation level.

      Creates a new FeatureChannelDataset for a single channel at a single
      decimation level. Input data can be provided as numpy array, xarray,
      DataFrame, or created empty.

      :param fc_name: Name for the Fourier Coefficient channel.
      :type fc_name: str
      :param fc_data: Input data. Can be numpy array (time, frequency) or xarray/DataFrame
                      format. Default is None (creates empty dataset).
      :type fc_data: np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, optional
      :param fc_metadata: Metadata for the channel. Default is None.
      :type fc_metadata: FeatureDecimationChannel, optional
      :param max_shape: Maximum shape for HDF5 dataset dimensions (expandable if None).
      :type max_shape: tuple, default=(None, None)
      :param chunks: Whether to use HDF5 chunking.
      :type chunks: bool, default=True
      :param dtype: Data type for the dataset (e.g., complex, float, int).
      :type dtype: type, default=complex
      :param \*\*kwargs: Additional keyword arguments for HDF5 dataset creation.

      :returns: Newly created FeatureChannelDataset object.
      :rtype: FeatureChannelDataset

      :raises TypeError: If fc_data type is not supported or metadata type mismatch.
      :raises RuntimeError or OSError: If channel already exists (will return existing channel).

      .. rubric:: Notes

      Data layout assumes (time, frequency) organization:

      - time index: window start times
      - frequency index: harmonic indices or float values
      - data: complex Fourier coefficients

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> metadata = FeatureDecimationChannel(name='Ex')

      Create from numpy array:

      >>> fc_data = np.random.randn(100, 256) + 1j * np.random.randn(100, 256)
      >>> channel = decimation.add_channel('Ex', fc_data=fc_data, fc_metadata=metadata)

      Create empty channel (expandable):

      >>> channel = decimation.add_channel('Ex', fc_metadata=metadata)



   .. py:method:: get_channel(fc_name: str) -> mth5.groups.FeatureChannelDataset

      Retrieve a Fourier Coefficient channel by name.

      :param fc_name: Name of the channel to retrieve.
      :type fc_name: str

      :returns: The requested FeatureChannelDataset object.
      :rtype: FeatureChannelDataset

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> channel = decimation.get_channel('Ex')
      >>> data = channel.to_numpy()



   .. py:method:: remove_channel(fc_name: str) -> None

      Remove a Fourier Coefficient channel from the decimation level.

      Deletes the channel from the HDF5 file. Note that this removes the
      reference but does not reduce file size.

      :param fc_name: Name of the channel to remove.
      :type fc_name: str

      :raises MTH5Error: If the channel does not exist.

      .. rubric:: Notes

      To reduce HDF5 file size, copy desired data to a new file.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> decimation.remove_channel('Ex')



   .. py:method:: update_metadata() -> None

      Update metadata from all channels in the decimation level.

      Scans all channels and updates the decimation-level metadata with
      aggregated information including time ranges and sampling rates.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> decimation.update_metadata()



   .. py:method:: add_weights(weight_name: str, weight_data: Optional[numpy.ndarray] = None, weight_metadata: Optional[object] = None, max_shape: tuple = (None, None, None), chunks: bool = True, **kwargs) -> None

      Add weight or masking data for Fourier Coefficients.

      Creates a dataset to store weights or masks for quality control,
      frequency band selection, or time window filtering.

      :param weight_name: Name for the weight dataset.
      :type weight_name: str
      :param weight_data: Weight values. Default is None.
      :type weight_data: np.ndarray, optional
      :param weight_metadata: Metadata for the weight dataset. Default is None.
      :type weight_metadata: optional
      :param max_shape: Maximum shape for expandable dimensions.
      :type max_shape: tuple, default=(None, None, None)
      :param chunks: Whether to use HDF5 chunking.
      :type chunks: bool, default=True
      :param \*\*kwargs: Additional keyword arguments for HDF5 dataset creation.

      .. rubric:: Notes

      Weight datasets can track:

      - weight_channel: Per-channel weights
      - weight_band: Per-frequency-band weights
      - weight_time: Per-time-window weights

      This method is a placeholder for future implementation.

      .. rubric:: Examples

      >>> decimation = FeatureDecimationGroup(h5_group)
      >>> decimation.add_weights('coherency_weights', weight_data=weights)



.. py:class:: MasterStationGroup(group: h5py.Group, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Collection helper for all stations in a survey.

   The group lives at ``/Survey/Stations`` and offers convenience accessors
   to add, fetch, or remove stations along with a summary table.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> mth5_obj = mth5.MTH5()
   >>> _ = mth5_obj.open_mth5("/tmp/example.mth5", mode="a")
   >>> stations = mth5_obj.stations_group
   >>> _ = stations.add_station("MT001")
   >>> stations.station_summary.head()  # doctest: +SKIP


   .. py:property:: station_summary
      :type: pandas.DataFrame


      Return a summary DataFrame of all stations in the file.

      :returns: Columns include ``station``, ``start``, ``end``, ``latitude``,
                and ``longitude``. Empty if no stations are present.
      :rtype: pandas.DataFrame

      .. rubric:: Notes

      Timestamps are parsed to pandas ``datetime64[ns]`` when possible.

      .. rubric:: Examples

      >>> summary = stations.station_summary
      >>> list(summary.columns)
      ['station', 'start', 'end', 'latitude', 'longitude']


   .. py:method:: add_station(station_name: str, station_metadata: mt_metadata.timeseries.Station | None = None) -> StationGroup

      Add or fetch a station group at ``/Survey/Stations/<name>``.

      :param station_name: Station identifier, typically matches ``metadata.id``.
      :type station_name: str
      :param station_metadata: Metadata container to seed the station attributes.
      :type station_metadata: mt_metadata.timeseries.Station, optional

      :returns: Convenience wrapper for the created or existing station.
      :rtype: StationGroup

      :raises ValueError: If ``station_name`` is empty.

      .. rubric:: Examples

      >>> station = stations.add_station("MT001")
      >>> station.metadata.id
      'MT001'



   .. py:method:: get_station(station_name: str) -> StationGroup

      Return an existing station by name.

      :param station_name: Name of the station to retrieve.
      :type station_name: str

      :returns: Wrapper for the requested station.
      :rtype: StationGroup

      :raises MTH5Error: If the station does not exist.

      .. rubric:: Examples

      >>> existing = stations.get_station("MT001")
      >>> existing.name
      'MT001'



   .. py:method:: remove_station(station_name: str) -> None

      Delete a station group reference from the file.

      :param station_name: Existing station name.
      :type station_name: str

      .. rubric:: Notes

      HDF5 deletion removes the reference only; underlying storage is not
      reclaimed.

      .. rubric:: Examples

      >>> stations.remove_station("MT001")



.. py:class:: StationGroup(group: h5py.Group, station_metadata: mt_metadata.timeseries.Station | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Utility wrapper for a single station at ``/Survey/Stations/<id>``.

   Station groups manage run collections, metadata propagation, and provide
   summary utilities for quick inspection.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> m5 = mth5.MTH5()
   >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
   >>> station = m5.stations_group.add_station("MT001")
   >>> _ = station.add_run("MT001a")
   >>> station.run_summary.shape[0] >= 1
   True


   .. py:method:: initialize_group(**kwargs: Any) -> None

      Create default subgroups and write metadata.

      :param \*\*kwargs: Additional attributes to set on the instance before initialization.

      .. rubric:: Examples

      >>> station.initialize_group()



   .. py:property:: master_station_group
      :type: MasterStationGroup


      Shortcut to the containing master station group.


   .. py:property:: transfer_functions_group
      :type: mth5.groups.TransferFunctionsGroup


      Convenience accessor for ``/Station/Transfer_Functions``.


   .. py:property:: fourier_coefficients_group
      :type: mth5.groups.MasterFCGroup


      Convenience accessor for ``/Station/Fourier_Coefficients``.


   .. py:property:: features_group
      :type: mth5.groups.MasterFeaturesGroup


      Convenience accessor for ``/Station/Features``.


   .. py:property:: survey_metadata
      :type: mt_metadata.timeseries.Survey


      Return survey metadata with this station appended.


   .. py:method:: metadata() -> mt_metadata.timeseries.Station

      Station metadata enriched with run information.



   .. py:property:: name
      :type: str



   .. py:property:: run_summary
      :type: pandas.DataFrame


      Return a summary of runs belonging to the station.

      :returns: Columns include ``id``, ``start``, ``end``, ``components``,
                ``measurement_type``, ``sample_rate``, and ``hdf5_reference``.
      :rtype: pandas.DataFrame

      .. rubric:: Notes

      Channel lists stored as byte arrays or JSON strings are normalized
      before summarization.

      .. rubric:: Examples

      >>> station.run_summary.head()  # doctest: +SKIP


   .. py:method:: make_run_name(alphabet: bool = False) -> str | None

      Generate the next run name using an alphabetic or numeric suffix.

      :param alphabet: If ``True`` use letters (``a``, ``b``, ...); otherwise use
                       numeric suffixes (``001``).
      :type alphabet: bool, default False

      :returns: Proposed run name or ``None`` if generation fails.
      :rtype: str or None

      .. rubric:: Examples

      >>> station.metadata.id = "MT001"
      >>> station.make_run_name()
      'MT001a'



   .. py:method:: locate_run(sample_rate: float, start: str | mt_metadata.common.mttime.MTime) -> pandas.DataFrame | None

      Locate runs matching a sample rate and start time.

      :param sample_rate: Sample rate in samples per second.
      :type sample_rate: float
      :param start: Start time string or ``MTime`` instance.
      :type start: str or MTime

      :returns: Matching rows from ``run_summary`` or ``None`` when no match exists.
      :rtype: pandas.DataFrame or None

      .. rubric:: Examples

      >>> station.locate_run(256.0, "2020-01-01T00:00:00")  # doctest: +SKIP



   .. py:method:: add_run(run_name: str, run_metadata: mt_metadata.timeseries.Run | None = None) -> mth5.groups.RunGroup

      Add a run under this station.

      :param run_name: Run identifier (for example ``id`` + suffix).
      :type run_name: str
      :param run_metadata: Metadata container to seed the run attributes.
      :type run_metadata: mt_metadata.timeseries.Run, optional

      :returns: Wrapper for the created or existing run.
      :rtype: RunGroup

      .. rubric:: Examples

      >>> run = station.add_run("MT001a")
      >>> run.metadata.id
      'MT001a'



   .. py:method:: get_run(run_name: str) -> mth5.groups.RunGroup

      Return a run by name.

      :param run_name: Existing run name.
      :type run_name: str

      :returns: Wrapper for the requested run.
      :rtype: RunGroup

      :raises MTH5Error: If the run does not exist.

      .. rubric:: Examples

      >>> existing_run = station.get_run("MT001a")
      >>> existing_run.name
      'MT001a'



   .. py:method:: remove_run(run_name: str) -> None

      Remove a run from this station.

      :param run_name: Existing run name.
      :type run_name: str

      .. rubric:: Notes

      Deleting removes the reference only; storage is not reclaimed.

      .. rubric:: Examples

      >>> station.remove_run("MT001a")



   .. py:method:: update_station_metadata() -> None

      Deprecated alias for :py:meth:`update_metadata`.

      :raises DeprecationWarning: Always raised to direct callers to ``update_metadata``.

      .. rubric:: Examples

      >>> station.update_station_metadata()  # doctest: +ELLIPSIS
      Traceback (most recent call last):
      ...
      DeprecationWarning: 'update_station_metadata' has been deprecated use 'update_metadata()'



   .. py:method:: update_metadata() -> None

      Synchronize station metadata from contained runs.

      .. rubric:: Notes

      The station ``time_period`` is set to the min/max of all runs, and
      ``channels_recorded`` combines all recorded components.

      .. rubric:: Examples

      >>> _ = station.update_metadata()
      >>> station.metadata.time_period.start  # doctest: +SKIP
      '2020-01-01T00:00:00'



.. py:class:: MasterSurveyGroup(group: h5py.Group, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Collection helper for surveys under ``Experiment/Surveys``.

   Provides helpers to add, fetch, or remove surveys and to summarize all
   channels in the experiment.

   .. rubric:: Examples

   >>> from mth5 import mth5
   >>> m5 = mth5.MTH5()
   >>> _ = m5.open_mth5("/tmp/example.mth5", mode="a")
   >>> surveys = m5.surveys_group
   >>> _ = surveys.add_survey("survey_01")
   >>> surveys.channel_summary.head()  # doctest: +SKIP


   .. py:property:: channel_summary
      :type: pandas.DataFrame


      Return a DataFrame summarizing all channels across surveys.

      :returns: Columns include survey, station, run, location, component,
                start/end, sample info, orientation, units, and HDF5 reference.
      :rtype: pandas.DataFrame

      .. rubric:: Examples

      >>> summary = surveys.channel_summary
      >>> set(summary.columns) >= {"survey", "station", "run", "component"}
      True


   .. py:method:: add_survey(survey_name: str, survey_metadata: mt_metadata.timeseries.Survey | None = None) -> SurveyGroup

      Add or fetch a survey at ``/Experiment/Surveys/<name>``.

      :param survey_name: Survey identifier; validated with ``validate_name``.
      :type survey_name: str
      :param survey_metadata: Metadata container used to seed the survey attributes.
      :type survey_metadata: Survey, optional

      :returns: Wrapper for the created or existing survey.
      :rtype: SurveyGroup

      :raises ValueError: If ``survey_name`` is empty.
      :raises MTH5Error: If the provided metadata id conflicts with the group name.

      .. rubric:: Examples

      >>> survey = surveys.add_survey("survey_01")
      >>> survey.metadata.id
      'survey_01'



   .. py:method:: get_survey(survey_name: str) -> SurveyGroup

      Return an existing survey by name.

      :param survey_name: Existing survey name.
      :type survey_name: str

      :returns: Wrapper for the requested survey.
      :rtype: SurveyGroup

      :raises MTH5Error: If the survey does not exist.

      .. rubric:: Examples

      >>> existing = surveys.get_survey("survey_01")
      >>> existing.metadata.id
      'survey_01'



   .. py:method:: remove_survey(survey_name: str) -> None

      Delete a survey reference from the file.

      :param survey_name: Existing survey name.
      :type survey_name: str

      .. rubric:: Notes

      HDF5 deletion removes the reference only; storage is not reclaimed.

      .. rubric:: Examples

      >>> surveys.remove_survey("survey_01")



.. py:class:: SurveyGroup(group: h5py.Group, survey_metadata: mt_metadata.timeseries.Survey | None = None, **kwargs: Any)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Wrapper for a single survey at ``Experiment/Surveys/<id>``.

   Handles survey-level metadata, child groups (stations, reports, filters,
   standards), and synchronization utilities.

   .. rubric:: Examples

   >>> survey = surveys.add_survey("survey_01")
   >>> survey.metadata.id
   'survey_01'


   .. py:method:: initialize_group(**kwargs: Any) -> None

      Create default subgroups and write survey metadata.

      :param \*\*kwargs: Additional attributes to set on the instance before initialization.

      .. rubric:: Examples

      >>> survey.initialize_group()



   .. py:method:: metadata() -> mt_metadata.timeseries.Survey

      Survey metadata enriched with station and filter information.



   .. py:method:: write_metadata() -> None

      Write HDF5 attributes from the survey metadata object.



   .. py:property:: stations_group
      :type: mth5.groups.MasterStationGroup



   .. py:property:: filters_group
      :type: mth5.groups.FiltersGroup


      Convenience accessor for ``/Survey/Filters`` group.


   .. py:property:: reports_group
      :type: mth5.groups.ReportsGroup


      Convenience accessor for ``/Survey/Reports`` group.


   .. py:property:: standards_group
      :type: mth5.groups.StandardsGroup


      Convenience accessor for ``/Survey/Standards`` group.


   .. py:method:: update_survey_metadata(survey_dict: dict[str, Any] | None = None) -> None

      Deprecated alias for :py:meth:`update_metadata`.

      :raises DeprecationWarning: Always raised to direct callers to ``update_metadata``.

      .. rubric:: Examples

      >>> survey.update_survey_metadata()  # doctest: +ELLIPSIS
      Traceback (most recent call last):
      ...
      DeprecationWarning: 'update_survey_metadata' has been deprecated use 'update_metadata()'



   .. py:method:: update_metadata(survey_dict: dict[str, Any] | None = None) -> None

      Synchronize survey metadata from station summaries.

      :param survey_dict: Additional metadata values to merge before synchronization.
      :type survey_dict: dict, optional

      .. rubric:: Notes

      Updates survey start/end dates and bounding box from station summaries,
      then writes metadata to HDF5.

      .. rubric:: Examples

      >>> _ = survey.update_metadata()
      >>> survey.metadata.time_period.start_date  # doctest: +SKIP
      '2020-01-01'



.. py:class:: ExperimentGroup(group, **kwargs)

   Bases: :py:obj:`mth5.groups.BaseGroup`


   Utility class to hold general information about the experiment and
   accompanying metadata for an MT experiment.

   To access the hdf5 group directly use `ExperimentGroup.hdf5_group`.

   >>> experiment = ExperimentGroup(hdf5_group)
   >>> experiment.hdf5_group.ref
   <HDF5 Group Reference>

   .. note:: All attributes should be input into the metadata object, that
            way all input will be validated against the metadata standards.
            If you change attributes in metadata object, you should run the
            `ExperimentGroup.write_metadata()` method.  This is a temporary
            solution, working on an automatic updater if metadata is changed.

   >>> experiment.metadata.existing_attribute = 'update_existing_attribute'
   >>> experiment.write_metadata()

   If you want to add a new attribute this should be done using the
   `metadata.add_base_attribute` method.

   >>> experiment.metadata.add_base_attribute('new_attribute',
   >>> ...                                'new_attribute_value',
   >>> ...                                {'type':str,
   >>> ...                                 'required':True,
   >>> ...                                 'style':'free form',
   >>> ...                                 'description': 'new attribute desc.',
   >>> ...                                 'units':None,
   >>> ...                                 'options':[],
   >>> ...                                 'alias':[],
   >>> ...                                 'example':'new attribute

   .. tip:: If you want ot add stations, reports, etc to the experiment this
             should be done from the MTH5 object.  This is to avoid
             duplication, at least for now.

   To look at what the structure of ``/Experiment`` looks like:

       >>> experiment
       /Experiment:
       ====================
           |- Group: Surveys
           -----------------
           |- Group: Reports
           -----------------
           |- Group: Standards
           -------------------
           |- Group: Stations
           ------------------



   .. py:method:: metadata()

      Overwrite get metadata to include station information



   .. py:property:: surveys_group


