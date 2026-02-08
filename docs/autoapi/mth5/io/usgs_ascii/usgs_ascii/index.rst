mth5.io.usgs_ascii.usgs_ascii
=============================

.. py:module:: mth5.io.usgs_ascii.usgs_ascii

.. autoapi-nested-parse::

   Created on Thu Aug 27 16:54:09 2020

   :author: Jared Peacock

   :license: MIT



Classes
-------

.. autoapisummary::

   mth5.io.usgs_ascii.usgs_ascii.USGSascii


Functions
---------

.. autoapisummary::

   mth5.io.usgs_ascii.usgs_ascii.read_ascii


Module Contents
---------------

.. py:class:: USGSascii(fn=None, **kwargs)

   Bases: :py:obj:`mth5.io.usgs_ascii.AsciiMetadata`


   Read and write USGS ascii formatted time series.

   =================== =======================================================
   Attributes          Description
   =================== =======================================================
   ts                  Pandas dataframe holding the time series data
   fn                  Full path to .asc file
   station_dir         Full path to station directory
   meta_notes          Notes of how the station was collected
   =================== =======================================================

   :Example: ::

       >>> zc = Z3DCollection()
       >>> fn_list = zc.get_time_blocks(z3d_path)
       >>> zm = USGSasc()
       >>> zm.SurveyID = 'iMUSH'
       >>> zm.get_z3d_db(fn_list[0])
       >>> zm.read_mtft24_cfg()
       >>> zm.CoordinateSystem = 'Geomagnetic North'
       >>> zm.SurveyID = 'MT'
       >>> zm.write_asc_file(str_fmt='%15.7e')
       >>> zm.write_station_info_metadata()



   .. py:attribute:: ts
      :value: None



   .. py:attribute:: station_dir


   .. py:attribute:: meta_notes
      :value: None



   .. py:property:: hx

      HX


   .. py:property:: hy

      hy


   .. py:property:: hz

      hz


   .. py:property:: ex

      ex


   .. py:property:: ey

      ey


   .. py:method:: to_run_ts()

      Get xarray for run



   .. py:method:: read(fn=None)

      Read in a USGS ascii file and fill attributes accordingly.

      :param fn: full path to .asc file to be read in
      :type fn: string



   .. py:method:: write(save_fn=None, chunk_size=1024, str_fmt='%15.7e', full=True, compress=False, save_dir=None, compress_type='zip', convert_electrics=True)

      Write an ascii file in the USGS ascii format.

      :param save_fn: full path to file name to save the merged ascii to
      :type save_fn: string

      :param chunck_size: chunck size to write file in blocks, larger numbers
                          are typically slower.
      :type chunck_size: int

      :param str_fmt: format of the data as written
      :type str_fmt: string

      :param full: write out the complete file, mostly for testing.
      :type full: boolean [ True | False ]

      :param compress: compress file
      :type compress: boolean [ True | False ]

      :param compress_type: compress file using zip or gzip
      :type compress_type: boolean [ zip | gzip ]



.. py:function:: read_ascii(fn)

   read USGS ASCII formatted file

   :param fn: DESCRIPTION
   :type fn: TYPE
   :return: DESCRIPTION
   :rtype: TYPE



