mth5.io.usgs_ascii.header
=========================

.. py:module:: mth5.io.usgs_ascii.header

.. autoapi-nested-parse::

   Created on Thu Aug 27 16:54:09 2020

   :author: Jared Peacock

   :license: MIT



Classes
-------

.. autoapisummary::

   mth5.io.usgs_ascii.header.AsciiMetadata


Module Contents
---------------

.. py:class:: AsciiMetadata(fn=None, **kwargs)

   Container for all the important metadata in a USGS ascii file.

   ========================= =================================================
   Attributes                Description
   ========================= =================================================
   survey_id                  Survey name
   site_id                    Site name
   run_id                     Run number
   site_latitude              Site latitude in decimal degrees WGS84
   site_longitude             Site longitude in decimal degrees WGS84
   site_elevation             Site elevation according to national map meters
   start                      Start time of station YYYY-MM-DDThh:mm:ss UTC
   end                        Stop time of station YYYY-MM-DDThh:mm:ss UTC
   sample_rate                Sampling rate samples/second
   n_samples                  Number of samples
   n_channels                 Number of channels
   coordinate_system          [ Geographic North | Geomagnetic North ]
   chn_settings               Channel settings, see below
   missing_data_flag          Missing data value
   ========================= =================================================

   :chn_settings:

   ========================= =================================================
   Keys                      Description
   ========================= =================================================
   ChnNum                    site_id+channel number
   ChnID                     Component [ ex | ey | hx | hy | hz ]
   InstrumentID              Data logger + sensor number
   Azimuth                   Setup angle of componet in degrees relative to
                             coordinate_system
   Dipole_Length             Dipole length in meters
   ========================= =================================================




   .. py:attribute:: logger


   .. py:property:: fn


   .. py:attribute:: missing_data_flag


   .. py:attribute:: coordinate_system
      :value: None



   .. py:attribute:: ex_metadata


   .. py:attribute:: ey_metadata


   .. py:attribute:: hx_metadata


   .. py:attribute:: hy_metadata


   .. py:attribute:: hz_metadata


   .. py:attribute:: channel_order
      :value: ['hx', 'ex', 'hy', 'ey', 'hz']



   .. py:attribute:: national_map_url
      :value: 'https://epqs.nationalmap.gov/v1/json?x={0:.5f}&y={1:.5f}&units=Meters&wkid=4326&includeDate=False'



   .. py:property:: file_size


   .. py:property:: survey_id


   .. py:property:: run_id


   .. py:property:: site_id


   .. py:property:: latitude


   .. py:property:: longitude


   .. py:property:: elevation

      get elevation from national map


   .. py:property:: start


   .. py:property:: end


   .. py:property:: n_channels


   .. py:property:: sample_rate


   .. py:property:: n_samples


   .. py:property:: survey_metadata


   .. py:property:: station_metadata


   .. py:property:: run_metadata


   .. py:method:: get_component_info(comp)

      :param comp: DESCRIPTION
      :type comp: TYPE
      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: read_metadata(fn=None, meta_lines=None)

      Read in a meta from the raw string or file.  Populate all metadata
      as attributes.

      :param fn: full path to USGS ascii file
      :type fn: string

      :param meta_lines: lines of metadata to read
      :type meta_lines: list



   .. py:method:: write_metadata(chn_list=['Ex', 'Ey', 'Hx', 'Hy', 'Hz'])

              Write out metadata in the format of USGS ascii.

              :return: list of metadate lines.

              .. note:: meant to use '
      '.join(lines) to write out in a file.





