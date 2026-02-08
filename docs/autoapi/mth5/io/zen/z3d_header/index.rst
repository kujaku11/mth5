mth5.io.zen.z3d_header
======================

.. py:module:: mth5.io.zen.z3d_header

.. autoapi-nested-parse::

   ====================
   Zen Header
   ====================

       * Tools for reading and writing files for Zen and processing software
       * Tools for copying data from SD cards
       * Tools for copying schedules to SD cards

   Created on Tue Jun 11 10:53:23 2013
   Updated August 2020 (JP)

   :copyright:
       Jared Peacock (jpeacock@usgs.gov)

   :license:
       MIT



Classes
-------

.. autoapisummary::

   mth5.io.zen.z3d_header.Z3DHeader


Module Contents
---------------

.. py:class:: Z3DHeader(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None, **kwargs: Any)

   Read header information from a Z3D file and make each metadata entry an attribute.

   :param fn: Full path to Z3D file.
   :type fn: str | pathlib.Path, optional
   :param fid: File object (e.g., open(Z3Dfile, 'rb')).
   :type fid: BinaryIO, optional
   :param \*\*kwargs: Additional keyword arguments to set as attributes.
   :type \*\*kwargs: dict

   .. attribute:: _header_len

      Length of header in bits (512).

      :type: int

   .. attribute:: ad_gain

      Gain of channel.

      :type: float | None

   .. attribute:: ad_rate

      Sampling rate in Hz.

      :type: float | None

   .. attribute:: alt

      Altitude of the station (not reliable).

      :type: float | None

   .. attribute:: attenchannelsmask

      Attenuation channels mask.

      :type: str | None

   .. attribute:: box_number

      ZEN box number.

      :type: float | None

   .. attribute:: box_serial

      ZEN box serial number.

      :type: str | None

   .. attribute:: channel

      Channel number of the file.

      :type: float | None

   .. attribute:: channelserial

      Serial number of the channel board.

      :type: str | None

   .. attribute:: ch_factor

      Channel factor (default 9.536743164062e-10).

      :type: float

   .. attribute:: channelgain

      Channel gain (default 1.0).

      :type: float

   .. attribute:: duty

      Duty cycle of the transmitter.

      :type: float | None

   .. attribute:: fid

      File object.

      :type: BinaryIO | None

   .. attribute:: fn

      Full path to Z3D file.

      :type: str | pathlib.Path | None

   .. attribute:: fpga_buildnum

      Build number of one of the boards.

      :type: float | None

   .. attribute:: gpsweek

      GPS week (default 1740).

      :type: int

   .. attribute:: header_str

      Full header string.

      :type: bytes | None

   .. attribute:: lat

      Latitude of station in degrees.

      :type: float | None

   .. attribute:: logterminal

      Log terminal setting.

      :type: str | None

   .. attribute:: long

      Longitude of the station in degrees.

      :type: float | None

   .. attribute:: main_hex_buildnum

      Build number of the ZEN box in hexadecimal.

      :type: float | None

   .. attribute:: numsats

      Number of GPS satellites.

      :type: float | None

   .. attribute:: old_version

      Whether this is an old version Z3D file (default False).

      :type: bool

   .. attribute:: period

      Period of the transmitter.

      :type: float | None

   .. attribute:: tx_duty

      Transmitter duty cycle.

      :type: float | None

   .. attribute:: tx_freq

      Transmitter frequency.

      :type: float | None

   .. attribute:: version

      Version of the firmware.

      :type: float | None

   .. rubric:: Examples

   >>> from mth5.io.zen import Z3DHeader
   >>> Z3Dfn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
   >>> header_obj = Z3DHeader(fn=Z3Dfn)
   >>> header_obj.read_header()


   .. py:attribute:: logger


   .. py:attribute:: fn
      :type:  str | pathlib.Path | None
      :value: None



   .. py:attribute:: fid
      :type:  BinaryIO | None
      :value: None



   .. py:attribute:: header_str
      :type:  bytes | None
      :value: None



   .. py:attribute:: ad_gain
      :type:  float | None
      :value: None



   .. py:attribute:: ad_rate
      :type:  float | None
      :value: None



   .. py:attribute:: alt
      :type:  float | None
      :value: None



   .. py:attribute:: attenchannelsmask
      :type:  str | None
      :value: None



   .. py:attribute:: box_number
      :type:  float | None
      :value: None



   .. py:attribute:: box_serial
      :type:  str | None
      :value: None



   .. py:attribute:: channel
      :type:  float | None
      :value: None



   .. py:attribute:: channelserial
      :type:  str | None
      :value: None



   .. py:attribute:: duty
      :type:  float | None
      :value: None



   .. py:attribute:: fpga_buildnum
      :type:  float | None
      :value: None



   .. py:attribute:: gpsweek
      :type:  int
      :value: 1740



   .. py:attribute:: lat
      :type:  float | None
      :value: None



   .. py:attribute:: logterminal
      :type:  str | None
      :value: None



   .. py:attribute:: long
      :type:  float | None
      :value: None



   .. py:attribute:: main_hex_buildnum
      :type:  float | None
      :value: None



   .. py:attribute:: numsats
      :type:  float | None
      :value: None



   .. py:attribute:: period
      :type:  float | None
      :value: None



   .. py:attribute:: tx_duty
      :type:  float | None
      :value: None



   .. py:attribute:: tx_freq
      :type:  float | None
      :value: None



   .. py:attribute:: version
      :type:  float | None
      :value: None



   .. py:attribute:: old_version
      :type:  bool
      :value: False



   .. py:attribute:: ch_factor
      :type:  float
      :value: 9.536743164062e-10



   .. py:attribute:: channelgain
      :type:  float
      :value: 1.0



   .. py:property:: data_logger
      :type: str


      Data logger name as ZEN{box_number}.

      :returns: Data logger name formatted as 'ZEN' followed by zero-padded box number.
      :rtype: str

      :raises TypeError: If box_number is None or cannot be converted to int.


   .. py:method:: read_header(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None) -> None

      Read the header information into appropriate attributes.

      Parses the header information from a Z3D file and populates the object's
      attributes with the extracted values. Supports both modern and legacy
      Z3D file formats.

      :param fn: Full path to Z3D file. If None, uses the instance's fn attribute.
      :type fn: str | pathlib.Path, optional
      :param fid: File object (e.g., open(Z3Dfile, 'rb')). If None, uses the instance's
                  fid attribute or opens the file specified by fn.
      :type fid: BinaryIO, optional

      :raises UnicodeDecodeError: If header bytes cannot be decoded as text.

      .. rubric:: Notes

      This method reads the first 512 bytes of the Z3D file as the header.
      It supports two formats:

      1. Modern format: key=value pairs separated by newlines
      2. Legacy format: comma-separated key:value pairs

      The method automatically detects legacy format and sets old_version=True.

      Coordinate values (lat/long) are automatically converted from radians
      to degrees, with validation to ensure they fall within valid ranges.

      .. rubric:: Examples

      >>> header_obj = Z3DHeader()
      >>> header_obj.read_header("/path/to/file.Z3D")

      >>> with open("/path/to/file.Z3D", "rb") as fid:
      ...     header_obj.read_header(fid=fid)



   .. py:method:: convert_value(key_string: str, value_string: str) -> float | str

      Convert the value to the appropriate units given the key.

      Converts string values to appropriate types based on the key name.
      Special handling is provided for latitude and longitude values, which
      are converted from radians to degrees with validation.

      :param key_string: The metadata key name, used to determine conversion type.
      :type key_string: str
      :param value_string: The string value to convert.
      :type value_string: str

      :returns: Converted value. Returns float for numeric values, str for
                non-numeric values. Latitude and longitude values are converted
                from radians to degrees.
      :rtype: float or str

      .. rubric:: Notes

      - Attempts to convert all values to float first
      - If conversion fails, returns original string
      - For keys containing 'lat', 'lon', or 'long':
        - Converts from radians to degrees using np.rad2deg
        - Validates latitude range (±90°), sets to 0.0 if invalid
        - Validates longitude range (±180°), sets to 0.0 if invalid

      .. rubric:: Examples

      >>> header = Z3DHeader()
      >>> header.convert_value("version", "4147")
      4147.0
      >>> header.convert_value("lat", "0.706816081")  # radians
      40.49757833327694  # degrees
      >>> header.convert_value("channelserial", "0xD474777C")
      '0xD474777C'



