mth5.groups.filters
===================

.. py:module:: mth5.groups.filters

.. autoapi-nested-parse::

   Filter groups manager for handling multiple filter types in MTH5.

   This module provides a unified interface for managing different types of filters
   including zeros-poles-gain (ZPK), coefficients, time delays, frequency-amplitude-phase (FAP),
   and finite impulse response (FIR) filters.

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license: MIT



Classes
-------

.. autoapisummary::

   mth5.groups.filters.FiltersGroup


Module Contents
---------------

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



