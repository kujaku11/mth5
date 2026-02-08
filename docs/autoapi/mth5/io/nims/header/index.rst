mth5.io.nims.header
===================

.. py:module:: mth5.io.nims.header

.. autoapi-nested-parse::

   Created on Thu Sep  1 12:57:32 2022

   @author: jpeacock



Exceptions
----------

.. autoapisummary::

   mth5.io.nims.header.NIMSError


Classes
-------

.. autoapisummary::

   mth5.io.nims.header.NIMSHeader


Module Contents
---------------

.. py:exception:: NIMSError

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.


.. py:class:: NIMSHeader(fn: Optional[Union[str, pathlib.Path]] = None)

   Class to hold NIMS header information.

   This class parses and stores header information from NIMS DATA.BIN files.
   The header contains metadata about the measurement site, equipment setup,
   GPS coordinates, electrode configuration, and other survey parameters.

   :param fn: Path to the NIMS file to read, by default None
   :type fn: str or Path, optional

   .. attribute:: fn

      Path to the NIMS file

      :type: Path or None

   .. attribute:: site_name

      Name of the measurement site

      :type: str or None

   .. attribute:: state_province

      State or province of the measurement location

      :type: str or None

   .. attribute:: country

      Country of the measurement location

      :type: str or None

   .. attribute:: box_id

      System box identifier

      :type: str or None

   .. attribute:: mag_id

      Magnetometer head identifier

      :type: str or None

   .. attribute:: ex_length

      North-South electric field wire length in meters

      :type: float or None

   .. attribute:: ex_azimuth

      North-South electric field wire heading in degrees

      :type: float or None

   .. attribute:: ey_length

      East-West electric field wire length in meters

      :type: float or None

   .. attribute:: ey_azimuth

      East-West electric field wire heading in degrees

      :type: float or None

   .. attribute:: n_electrode_id

      North electrode identifier

      :type: str or None

   .. attribute:: s_electrode_id

      South electrode identifier

      :type: str or None

   .. attribute:: e_electrode_id

      East electrode identifier

      :type: str or None

   .. attribute:: w_electrode_id

      West electrode identifier

      :type: str or None

   .. attribute:: ground_electrode_info

      Ground electrode information

      :type: str or None

   .. attribute:: header_gps_stamp

      GPS timestamp from header

      :type: MTime or None

   .. attribute:: header_gps_latitude

      GPS latitude from header in decimal degrees

      :type: float or None

   .. attribute:: header_gps_longitude

      GPS longitude from header in decimal degrees

      :type: float or None

   .. attribute:: header_gps_elevation

      GPS elevation from header in meters

      :type: float or None

   .. attribute:: operator

      Operator name

      :type: str or None

   .. attribute:: comments

      Survey comments

      :type: str or None

   .. attribute:: run_id

      Run identifier

      :type: str or None

   .. attribute:: data_start_seek

      Byte position where data begins in file

      :type: int

   .. rubric:: Examples

   A typical header looks like:

   .. code-block::

       '''
       >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       >>>user field>>>>>>>>>>>>>>>>>>>>>>>>>>>>
       SITE NAME: Budwieser Spring
       STATE/PROVINCE: CA
       COUNTRY: USA
       >>> The following code in double quotes is REQUIRED to start the NIMS <<
       >>> The next 3 lines contain values required for processing <<<<<<<<<<<<
       >>> The lines after that are optional <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       "300b"  <-- 2CHAR EXPERIMENT CODE + 3 CHAR SITE CODE + RUN LETTER
       1105-3; 1305-3  <-- SYSTEM BOX I.D.; MAG HEAD ID (if different)
       106  0 <-- N-S Ex WIRE LENGTH (m); HEADING (deg E mag N)
       109  90 <-- E-W Ey WIRE LENGTH (m); HEADING (deg E mag N)
       1         <-- N ELECTRODE ID
       3         <-- E ELECTRODE ID
       2         <-- S ELECTRODE ID
       4         <-- W ELECTRODE ID
       Cu        <-- GROUND ELECTRODE INFO
       GPS INFO: 26/09/19 18:29:29 34.7268 N 115.7350 W 939.8
       OPERATOR: KP
       COMMENT: N/S CRS: .95/.96 DCV: 3.5 ACV:1
       E/W CRS: .85/.86 DCV: 1.5 ACV: 1
       Redeployed site for run b b/c possible animal disturbance
       '''


   .. py:attribute:: logger


   .. py:property:: fn
      :type: Optional[pathlib.Path]


      Full path to NIMS file.

      :returns: Path object representing the NIMS file location,
                or None if no file is set
      :rtype: Path or None


   .. py:attribute:: header_dict
      :value: None



   .. py:attribute:: site_name
      :value: None



   .. py:attribute:: state_province
      :value: None



   .. py:attribute:: country
      :value: None



   .. py:attribute:: box_id
      :value: None



   .. py:attribute:: mag_id
      :value: None



   .. py:attribute:: ex_length
      :value: None



   .. py:attribute:: ex_azimuth
      :value: None



   .. py:attribute:: ey_length
      :value: None



   .. py:attribute:: ey_azimuth
      :value: None



   .. py:attribute:: n_electrode_id
      :value: None



   .. py:attribute:: s_electrode_id
      :value: None



   .. py:attribute:: e_electrode_id
      :value: None



   .. py:attribute:: w_electrode_id
      :value: None



   .. py:attribute:: ground_electrode_info
      :value: None



   .. py:attribute:: header_gps_stamp
      :value: None



   .. py:attribute:: header_gps_latitude
      :value: None



   .. py:attribute:: header_gps_longitude
      :value: None



   .. py:attribute:: header_gps_elevation
      :value: None



   .. py:attribute:: operator
      :value: None



   .. py:attribute:: comments


   .. py:attribute:: run_id
      :value: None



   .. py:attribute:: data_start_seek
      :value: 0



   .. py:property:: station
      :type: Optional[str]


      Station ID derived from run ID.

      :returns: Station identifier (run ID without the last character),
                or None if run_id is not set
      :rtype: str or None

      .. rubric:: Notes

      The station ID is typically the run ID with the last character
      (run letter) removed.


   .. py:property:: file_size
      :type: Optional[int]


      Size of the NIMS file in bytes.

      :returns: File size in bytes, or None if no file is set
      :rtype: int or None

      :raises FileNotFoundError: If the file does not exist


   .. py:method:: read_header(fn: Optional[Union[str, pathlib.Path]] = None) -> None

      Read header information from a NIMS file.

      This method reads and parses the header section of a NIMS DATA.BIN file,
      extracting metadata about the survey setup, GPS coordinates, electrode
      configuration, and other parameters.

      :param fn: Full path to NIMS file to read. Uses self.fn if not provided.
      :type fn: str or Path, optional

      :raises NIMSError: If the file does not exist or cannot be read

      .. rubric:: Notes

      The method reads up to _max_header_length bytes from the beginning
      of the file, parses the header information, and stores the results
      in the header_dict attribute and individual properties.



   .. py:method:: parse_header_dict(header_dict: Optional[dict[str, str]] = None) -> None

      Parse the header dictionary into individual attributes.

      This method takes the raw header dictionary and extracts specific
      information into class attributes for easy access.

      :param header_dict: Dictionary containing header key-value pairs. Uses self.header_dict
                          if not provided.
      :type header_dict: dict of str, optional

      .. rubric:: Notes

      Parses various header fields including:
      - Wire lengths and azimuths for electric field measurements
      - System box and magnetometer IDs
      - GPS coordinates and timestamp
      - Run identifier
      - Other metadata fields



