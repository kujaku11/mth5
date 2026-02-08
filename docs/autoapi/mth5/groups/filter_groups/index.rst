mth5.groups.filter_groups
=========================

.. py:module:: mth5.groups.filter_groups

.. autoapi-nested-parse::

   Import all Group objects



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/groups/filter_groups/coefficient_filter_group/index
   /autoapi/mth5/groups/filter_groups/fap_filter_group/index
   /autoapi/mth5/groups/filter_groups/fir_filter_group/index
   /autoapi/mth5/groups/filter_groups/time_delay_filter_group/index
   /autoapi/mth5/groups/filter_groups/zpk_filter_group/index


Classes
-------

.. autoapisummary::

   mth5.groups.filter_groups.CoefficientGroup
   mth5.groups.filter_groups.TimeDelayGroup
   mth5.groups.filter_groups.ZPKGroup
   mth5.groups.filter_groups.FAPGroup
   mth5.groups.filter_groups.FIRGroup


Package Contents
----------------

.. py:class:: CoefficientGroup(group, **kwargs)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Container for Coefficient type filters


   .. py:property:: filter_dict

      Dictionary of available coefficient filters

      :return: DESCRIPTION
      :rtype: TYPE


   .. py:method:: add_filter(name, coefficient_metadata)

      Add a coefficient Filter

      :param name: DESCRIPTION
      :type name: TYPE
      :param coefficient_metadata: DESCRIPTION
      :type coefficient_metadata: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: remove_filter()


   .. py:method:: get_filter(name)

      Get a filter from the name

      :param name: name of the filter
      :type name: string

      :return: HDF5 group of the ZPK filter



   .. py:method:: from_object(coefficient_object)

      make a filter from a :class:`mt_metadata.timeseries.filters.CoefficientFilter`

      :param zpk_object: MT metadata Coefficient Filter
      :type zpk_object: :class:`mt_metadata.timeseries.filters.CoefficientFilter`




   .. py:method:: to_object(name)

      make a :class:`mt_metadata.timeseries.filters.CoefficientFilter` object

      :return: DESCRIPTION
      :rtype: TYPE




.. py:class:: TimeDelayGroup(group, **kwargs)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Container for time_delay type filters



   .. py:property:: filter_dict

      Dictionary of available time_delay filters

      :return: DESCRIPTION
      :rtype: TYPE


   .. py:method:: add_filter(name, time_delay_metadata)

      create an HDF5 group/dataset from information given.

      :param name: Nane of the filter
      :type name: string
      :param poles: poles of the filter as complex numbers
      :type poles: np.ndarray(dtype=complex)
      :param zeros: zeros of the filter as complex numbers
      :type zeros: np.ndarray(dtype=comples)
      :param time_delay_metadata: metadata dictionary see
      :class:`mt_metadata.timeseries.filters.PoleZeroFilter` for details on entries
      :type time_delay_metadata: dictionary




   .. py:method:: remove_filter()


   .. py:method:: get_filter(name)

      Get a filter from the name

      :param name: name of the filter
      :type name: string

      :return: HDF5 group of the time_delay filter



   .. py:method:: from_object(time_delay_object)

      make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

      :param time_delay_object: MT metadata PoleZeroFilter
      :type time_delay_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`




   .. py:method:: to_object(name)

      make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object

      :return: DESCRIPTION
      :rtype: TYPE




.. py:class:: ZPKGroup(group, **kwargs)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Container for ZPK type filters



   .. py:property:: filter_dict

      Dictionary of available ZPK filters

      :return: DESCRIPTION
      :rtype: TYPE


   .. py:method:: add_filter(name, poles, zeros, zpk_metadata)

      create an HDF5 group/dataset from information given.

      :param name: Nane of the filter
      :type name: string
      :param poles: poles of the filter as complex numbers
      :type poles: np.ndarray(dtype=complex)
      :param zeros: zeros of the filter as complex numbers
      :type zeros: np.ndarray(dtype=comples)
      :param zpk_metadata: metadata dictionary see
      :class:`mt_metadata.timeseries.filters.PoleZeroFilter` for details on entries
      :type zpk_metadata: dictionary




   .. py:method:: remove_filter()


   .. py:method:: get_filter(name)

      Get a filter from the name

      :param name: name of the filter
      :type name: string

      :return: HDF5 group of the ZPK filter



   .. py:method:: from_object(zpk_object)

      make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

      :param zpk_object: MT metadata PoleZeroFilter
      :type zpk_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`




   .. py:method:: to_object(name)

      make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object

      :return: DESCRIPTION
      :rtype: TYPE




.. py:class:: FAPGroup(group, **kwargs)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Container for fap type filters



   .. py:property:: filter_dict

      Dictionary of available fap filters

      :return: DESCRIPTION
      :rtype: TYPE


   .. py:method:: add_filter(name, frequency, amplitude, phase, fap_metadata)

      create an HDF5 group/dataset from information given.

      :param name: name of the filter
      :type name: string
      :param frequency: frequency array in samples per second
      :type frequency: list, np.ndarray
      :param amplitude: amplitude array in units of units out
      :type amplitude: list, np.ndarray
      :param phase: Phase in degrees
      :type phase: list, np.ndarray
      :param fap_metadata: other metadata for the filter see
      :class:`mt_metadata.timeseries.filters.FrequencyResponseTableFilter`          for details on entries
      :type fap_metadata: dictionary
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: remove_filter()


   .. py:method:: get_filter(name)

      Get a filter from the name

      :param name: name of the filter
      :type name: string

      :return: HDF5 group of the fap filter



   .. py:method:: update_filter(fap_object)

      update values from fap object

      :param fap_object: DESCRIPTION
      :type fap_object: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: from_object(fap_object)

      make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

      :param fap_object: MT metadata PoleZeroFilter
      :type fap_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`




   .. py:method:: to_object(name)

      make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object

      :return: DESCRIPTION
      :rtype: TYPE




.. py:class:: FIRGroup(group, **kwargs)

   Bases: :py:obj:`mth5.groups.base.BaseGroup`


   Container for fir type filters



   .. py:property:: filter_dict

      Dictionary of available fir filters

      :return: DESCRIPTION
      :rtype: TYPE


   .. py:method:: add_filter(name, coefficients, fir_metadata)

      create an HDF5 group/dataset from information given.

      :param name: Nane of the filter
      :type name: string
      :param poles: poles of the filter as complex numbers
      :type poles: np.ndarray(dtype=complex)
      :param zeros: zeros of the filter as complex numbers
      :type zeros: np.ndarray(dtype=comples)
      :param fir_metadata: metadata dictionary see
      :class:`mt_metadata.timeseries.filters.PoleZeroFilter` for details on entries
      :type fir_metadata: dictionary




   .. py:method:: remove_filter()


   .. py:method:: get_filter(name)

      Get a filter from the name

      :param name: name of the filter
      :type name: string

      :return: HDF5 group of the fir filter



   .. py:method:: from_object(fir_object)

      make a filter from a :class:`mt_metadata.timeseries.filters.PoleZeroFilter`

      :param fir_object: MT metadata PoleZeroFilter
      :type fir_object: :class:`mt_metadata.timeseries.filters.PoleZeroFilter`




   .. py:method:: to_object(name)

      make a :class:`mt_metadata.timeseries.filters.pole_zeros_filter` object

      :return: DESCRIPTION
      :rtype: TYPE




