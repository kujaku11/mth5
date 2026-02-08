mth5.groups.filter_groups.coefficient_filter_group
==================================================

.. py:module:: mth5.groups.filter_groups.coefficient_filter_group

.. autoapi-nested-parse::

   Created on Wed Jun  9 08:58:15 2021

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license: MIT



Classes
-------

.. autoapisummary::

   mth5.groups.filter_groups.coefficient_filter_group.CoefficientGroup


Module Contents
---------------

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




