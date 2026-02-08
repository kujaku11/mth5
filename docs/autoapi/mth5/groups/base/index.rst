mth5.groups.base
================

.. py:module:: mth5.groups.base

.. autoapi-nested-parse::

   Base Group Class
   ====================

   Contains all the base functions that will be used by group classes.

   Created on Fri May 29 15:09:48 2020

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license:
       MIT



Attributes
----------

.. autoapisummary::

   mth5.groups.base.meta_classes


Classes
-------

.. autoapisummary::

   mth5.groups.base.BaseGroup


Module Contents
---------------

.. py:data:: meta_classes

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



