mth5.io.nims.response_filters
=============================

.. py:module:: mth5.io.nims.response_filters

.. autoapi-nested-parse::

   Created on Fri Sep  2 13:50:51 2022

   @author: jpeacock



Exceptions
----------

.. autoapisummary::

   mth5.io.nims.response_filters.ResponseError


Classes
-------

.. autoapisummary::

   mth5.io.nims.response_filters.Response


Module Contents
---------------

.. py:exception:: ResponseError

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.


.. py:class:: Response(system_id=None, **kwargs)

   Bases: :py:obj:`object`


   Common NIMS response filters for electric and magnetic channels



   .. py:attribute:: system_id
      :value: None



   .. py:attribute:: hardware
      :value: 'PC'



   .. py:attribute:: instrument_type
      :value: 'backbone'



   .. py:attribute:: sample_rate
      :value: 1



   .. py:attribute:: e_conversion_factor
      :value: 409600000.0



   .. py:attribute:: h_conversion_factor
      :value: 100



   .. py:attribute:: time_delays_dict


   .. py:property:: magnetic_low_pass

      Low pass 3 pole filter

      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: magnetic_conversion

      DESCRIPTION
      :rtype: TYPE

      :type: return


   .. py:property:: electric_low_pass

      5 pole electric low pass filter
      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: electric_high_pass_pc

      1-pole low pass filter for 8 hz instruments
      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: electric_high_pass_hp

      1-pole low pass for 1 hz instuments
      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: electric_conversion

      electric channel conversion from counts to Volts
      :return: DESCRIPTION
      :rtype: TYPE


   .. py:property:: electric_physical_units

      DESCRIPTION
      :rtype: TYPE

      :type: return


   .. py:method:: get_electric_high_pass(hardware='pc')

      get the electric high pass filter based on the hardware



   .. py:method:: dipole_filter(length)

      Make a dipole filter

      :param length: dipole length in meters
      :type length: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_channel_response(channel, dipole_length=1)

      Get the full channel response filter
      :param channel: DESCRIPTION
      :type channel: TYPE
      :param dipole_length: DESCRIPTION, defaults to 1
      :type dipole_length: TYPE, optional
      :return: DESCRIPTION
      :rtype: TYPE




