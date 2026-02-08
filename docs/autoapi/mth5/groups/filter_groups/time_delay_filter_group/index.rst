mth5.groups.filter_groups.time_delay_filter_group
=================================================

.. py:module:: mth5.groups.filter_groups.time_delay_filter_group

.. autoapi-nested-parse::

   Created on Wed Jun  9 09:01:55 2021

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license: MIT



Classes
-------

.. autoapisummary::

   mth5.groups.filter_groups.time_delay_filter_group.TimeDelayGroup


Module Contents
---------------

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




