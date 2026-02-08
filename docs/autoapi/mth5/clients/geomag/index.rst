mth5.clients.geomag
===================

.. py:module:: mth5.clients.geomag

.. autoapi-nested-parse::

   Created on Mon Nov 14 13:58:44 2022

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.clients.geomag.GeomagClient
   mth5.clients.geomag.USGSGeomag


Module Contents
---------------

.. py:class:: GeomagClient(**kwargs)

   Get geomagnetic data from observatories.

   key words

   - **observatory**: Geogmangetic observatory ID
   - **type**: type of data to get 'adjusted'
   - **start**: start date time to request UTC
   - **end**: end date time to request UTC
   - **elements**: components to get
   - **sampling_period**: samples between measurements in seconds
   - **format**: JSON or IAGA2002

   .. seealso:: https://www.usgs.gov/tools/web-service-geomagnetism-data



   .. py:attribute:: type
      :value: 'adjusted'



   .. py:property:: sampling_period


   .. py:property:: elements


   .. py:attribute:: format
      :value: 'json'



   .. py:property:: observatory


   .. py:property:: start


   .. py:property:: end


   .. py:property:: user_agent

      User agent for the URL request

      :return: DESCRIPTION
      :rtype: TYPE


   .. py:method:: get_chunks()

      Get the number of chunks of allowable sized to request, includes the elements

      So the max length is the maximum time period that can be requested but includes
      the number of elements in the request.  So if the max length is 172800 seconds
      and the sampling period is 1 second, then the maximum number of elements that can
      be requested is 172800 / (1 * len(elements)).

      :return: DESCRIPTION
      :rtype: TYPE




   .. py:method:: get_data(run_id='001')

      Get data from geomag client at USGS based on the request.  This might
      have to be done in chunks depending on the request size.  The returned
      output is a json object, which we should turn into a ChannelTS object

      For now read into a pandas dataframe and then into a ChannelTS

      In the future, if the request is large, think about writing
      directly to an MTH5 for better efficiency.

      :return: DESCRIPTION
      :rtype: TYPE




.. py:class:: USGSGeomag(**kwargs)

   .. py:attribute:: save_path


   .. py:attribute:: mth5_filename
      :value: None



   .. py:attribute:: interact
      :value: False



   .. py:attribute:: request_columns
      :value: ['observatory', 'type', 'elements', 'sampling_period', 'start', 'end']



   .. py:attribute:: h5_compression
      :value: 'gzip'



   .. py:attribute:: h5_compression_opts
      :value: 4



   .. py:attribute:: h5_shuffle
      :value: True



   .. py:attribute:: h5_fletcher32
      :value: True



   .. py:attribute:: h5_data_level
      :value: 1



   .. py:attribute:: mth5_file_mode
      :value: 'w'



   .. py:attribute:: mth5_version
      :value: '0.2.0'



   .. py:property:: h5_kwargs


   .. py:method:: validate_request_df(request_df)

      Make sure the input request dataframe has the appropriate columns

      :param request_df: request dataframe
      :type request_df: :class:`pandas.DataFrame`
      :return: valid request dataframe
      :rtype: :class:`pandas.DataFrame`




   .. py:method:: add_run_id(request_df)

      Add run id to request df

      :param request_df: request dataframe
      :type request_df: :class:`pandas.DataFrame`
      :return: add a run number to unique time windows for each observatory
       at each unique sampling period.
      :rtype: :class:`pandas.DataFrame`




   .. py:method:: make_mth5_from_geomag(request_df)

      Download geomagnetic observatory data from USGS webservices into an
      MTH5 using a request dataframe or csv file.

      :param request_df: DataFrame with columns

          - 'observatory'     --> Observatory code
          - 'type'            --> data type [ 'variation' | 'adjusted' | 'quasi-definitive' | 'definitive' ]
          - 'elements'        --> Elements to get [D, DIST, DST, E, E-E, E-N, F, G, H, SQ, SV, UK1, UK2, UK3, UK4, X, Y, Z]
          - 'sampling_period' --> sample period [ 1 | 60 | 3600 ]
          - 'start'           --> Start time YYYY-MM-DDThh:mm:ss
          - 'end'             --> End time YYYY-MM-DDThh:mm:ss

      :type request_df: :class:`pandas.DataFrame`, str or Path if csv file


      :return: if interact is True an MTH5 object is returned otherwise the
       path to the file is returned
      :rtype: Path or :class:`mth5.mth5.MTH5`

      .. seealso:: https://www.usgs.gov/tools/web-service-geomagnetism-data




