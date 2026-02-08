mth5.groups.filter_groups.fap_filter_group
==========================================

.. py:module:: mth5.groups.filter_groups.fap_filter_group

.. autoapi-nested-parse::

   Created on Wed Jun  9 08:55:16 2021

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license: MIT



Classes
-------

.. autoapisummary::

   mth5.groups.filter_groups.fap_filter_group.FAPGroup


Module Contents
---------------

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




