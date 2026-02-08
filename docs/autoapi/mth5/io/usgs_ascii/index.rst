mth5.io.usgs_ascii
==================

.. py:module:: mth5.io.usgs_ascii


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/usgs_ascii/header/index
   /autoapi/mth5/io/usgs_ascii/usgs_ascii/index
   /autoapi/mth5/io/usgs_ascii/usgs_ascii_collection/index


Classes
-------

.. autoapisummary::

   mth5.io.usgs_ascii.AsciiMetadata
   mth5.io.usgs_ascii.USGSasciiCollection


Package Contents
----------------

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





.. py:class:: USGSasciiCollection(file_path=None, **kwargs)

   Bases: :py:obj:`mth5.io.collection.Collection`


   Collection of USGS ASCII files.

   .. code-block:: python

       >>> from mth5.io.usgs_ascii import USGSasciiCollection
       >>> lc = USGSasciiCollection(r"/path/to/ascii/files")
       >>> run_dict = lc.get_runs(1)




   .. py:attribute:: file_ext
      :value: 'asc'



   .. py:method:: to_dataframe(sample_rates=[4], run_name_zeros=4, calibration_path=None)

      Create a data frame of each TXT file in a given directory.

      .. note:: If a run name is already present it will not be overwritten

      :param sample_rates: sample rate to get, defaults to [4]
      :type sample_rates: int or list, optional
      :param run_name_zeros: number of zeros to assing to the run name,
       defaults to 4
      :type run_name_zeros: int, optional
      :param calibration_path: path to calibration files, defaults to None
      :type calibration_path: string or Path, optional
      :return: Dataframe with information of each TXT file in the given
       directory.
      :rtype: :class:`pandas.DataFrame`

      :Example:

          >>> from mth5.io.usgs_ascii import USGSasciiCollection
          >>> lc = USGSasciiCollection("/path/to/ascii/files")
          >>> ascii_df = lc.to_dataframe()




   .. py:method:: assign_run_names(df, zeros=4)

      Assign run names based on start and end times, checks if a file has
      the same start time as the last end time.

      Run names are assigned as sr{sample_rate}_{run_number:0{zeros}}. Only
      if the run name is not assigned already.

      :param df: Dataframe with the appropriate columns
      :type df: :class:`pandas.DataFrame`
      :param zeros: number of zeros in run name, defaults to 4
      :type zeros: int, optional
      :return: Dataframe with run names
      :rtype: :class:`pandas.DataFrame`




