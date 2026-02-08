mth5.io.zen.z3d_schedule
========================

.. py:module:: mth5.io.zen.z3d_schedule

.. autoapi-nested-parse::

   Z3D Schedule File Parser

   This module provides functionality for parsing schedule information from Z3D files.
   The Z3DSchedule class extracts and processes schedule metadata stored at offset 512
   in Z3D files, providing access to various recording parameters and timing information.

   Created on Wed Aug 24 11:24:57 2022
   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.zen.z3d_schedule.Z3DSchedule


Module Contents
---------------

.. py:class:: Z3DSchedule(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None, **kwargs: Any)

   Parser for Z3D file schedule information and metadata.

   Reads schedule information from Z3D files and creates object attributes
   for each metadata entry. Schedule data is stored at byte offset 512 in
   Z3D files and contains recording parameters, timing information, and
   instrument configuration settings.

   The class preserves the original capitalization from the Z3D file format
   and provides automatic parsing of key-value pairs in the schedule section.

   :param fn: Full path to Z3D file to read schedule information from.
              Can be string path or pathlib.Path object.
   :type fn: str | pathlib.Path, optional
   :param fid: Open file object for reading Z3D file in binary mode.
               Example: open('file.z3d', 'rb')
   :type fid: BinaryIO, optional
   :param \*\*kwargs: Additional keyword arguments to set as object attributes.
   :type \*\*kwargs: Any

   .. attribute:: AutoGain

      Auto gain setting for the recording channel ['Y' or 'N'].

      :type: str | None

   .. attribute:: Comment

      User comments or notes for the schedule configuration.

      :type: str | None

   .. attribute:: Date

      Date when the schedule action was started in YYYY-MM-DD format.

      :type: str | None

   .. attribute:: Duty

      Duty cycle percentage of the transmitter (0-100).

      :type: str | None

   .. attribute:: FFTStacks

      Number of FFT stacks used by the transmitter.

      :type: str | None

   .. attribute:: Filename

      Original filename that the ZEN instrument assigns to the recording.

      :type: str | None

   .. attribute:: Gain

      Gain setting for the recording channel (e.g., '1.0000').

      :type: str | None

   .. attribute:: Log

      Data logging enabled flag ['Y' or 'N'].

      :type: str | None

   .. attribute:: NewFile

      Create new file for recording flag ['Y' or 'N'].

      :type: str | None

   .. attribute:: Period

      Base period setting for the transmitter in seconds.

      :type: str | None

   .. attribute:: RadioOn

      Radio communication enabled flag ['Y', 'N', or 'X'].

      :type: str | None

   .. attribute:: SR

      Sampling rate in Hz (originally 'S/R' in file, converted to 'SR').

      :type: str | None

   .. attribute:: SamplesPerAcq

      Number of samples per acquisition for transmitter mode.

      :type: str | None

   .. attribute:: Sleep

      Sleep mode enabled flag ['Y' or 'N'].

      :type: str | None

   .. attribute:: Sync

      GPS synchronization enabled flag ['Y' or 'N'].

      :type: str | None

   .. attribute:: Time

      Time when the schedule action started in HH:MM:SS format (GPS time).

      :type: str | None

   .. attribute:: initial_start

      Parsed start time as MTime object with GPS time flag enabled.
      Combines Date and Time attributes for timestamp calculation.

      :type: MTime

   .. attribute:: fn

      Path to the Z3D file being processed.

      :type: str | pathlib.Path | None

   .. attribute:: fid

      File object for reading the Z3D file.

      :type: BinaryIO | None

   .. attribute:: meta_string

      Raw schedule metadata string read from the file.

      :type: bytes | None

   .. attribute:: _header_len

      Length of Z3D file header in bytes (512).

      :type: int

   .. attribute:: _schedule_metadata_len

      Length of schedule metadata section in bytes (512).

      :type: int

   .. attribute:: logger

      Loguru logger instance for debugging and status messages.

      :type: Logger

   .. rubric:: Notes

   The Z3D file format stores schedule information at a fixed offset of 512 bytes
   from the beginning of the file. The schedule section is also 512 bytes long
   and contains key-value pairs in the format "Schedule.key = value".

   All schedule values are stored as strings to preserve the original format
   from the Z3D file. Boolean-like values use 'Y'/'N' convention.

   The initial_start attribute automatically converts Date and Time into a
   GPS-corrected MTime object for accurate timestamp handling.

   .. rubric:: Examples

   Read schedule from file path:

   >>> from mth5.io.zen import Z3DSchedule
   >>> from pathlib import Path
   >>>
   >>> # Using filename
   >>> schedule = Z3DSchedule(fn="recording.z3d")
   >>> schedule.read_schedule()
   >>> print(f"Sampling rate: {schedule.SR} Hz")
   >>> print(f"Start time: {schedule.initial_start}")

   Read schedule from file object:

   >>> with open("recording.z3d", "rb") as fid:
   ...     schedule = Z3DSchedule(fid=fid)
   ...     schedule.read_schedule()
   ...     print(f"Date: {schedule.Date}, Time: {schedule.Time}")
   ...     print(f"GPS Sync: {schedule.Sync}")

   Access schedule attributes:

   >>> schedule = Z3DSchedule()
   >>> schedule.read_schedule(fn="recording.z3d")
   >>>
   >>> # Check recording configuration
   >>> if schedule.Sync == 'Y':
   ...     print("GPS synchronization enabled")
   >>> if schedule.Log == 'Y':
   ...     print("Data logging enabled")
   >>>
   >>> # Get numeric values (stored as strings)
   >>> sample_rate = float(schedule.SR) if schedule.SR else 0
   >>> gain_value = float(schedule.Gain) if schedule.Gain else 1.0


   .. py:attribute:: logger


   .. py:attribute:: fn
      :type:  str | pathlib.Path | None
      :value: None



   .. py:attribute:: fid
      :type:  BinaryIO | None
      :value: None



   .. py:attribute:: meta_string
      :type:  bytes | None
      :value: None



   .. py:attribute:: AutoGain
      :type:  str | None
      :value: None



   .. py:attribute:: Comment
      :type:  str | None
      :value: None



   .. py:attribute:: Date
      :type:  str | None
      :value: None



   .. py:attribute:: Duty
      :type:  str | None
      :value: None



   .. py:attribute:: FFTStacks
      :type:  str | None
      :value: None



   .. py:attribute:: Filename
      :type:  str | None
      :value: None



   .. py:attribute:: Gain
      :type:  str | None
      :value: None



   .. py:attribute:: Log
      :type:  str | None
      :value: None



   .. py:attribute:: NewFile
      :type:  str | None
      :value: None



   .. py:attribute:: Period
      :type:  str | None
      :value: None



   .. py:attribute:: RadioOn
      :type:  str | None
      :value: None



   .. py:attribute:: SR
      :type:  str | None
      :value: None



   .. py:attribute:: SamplesPerAcq
      :type:  str | None
      :value: None



   .. py:attribute:: Sleep
      :type:  str | None
      :value: None



   .. py:attribute:: Sync
      :type:  str | None
      :value: None



   .. py:attribute:: Time
      :type:  str | None
      :value: None



   .. py:attribute:: initial_start
      :type:  mt_metadata.common.MTime


   .. py:method:: read_schedule(fn: str | pathlib.Path | None = None, fid: BinaryIO | None = None) -> None

      Read and parse schedule metadata from Z3D file.

      Reads the schedule information section from a Z3D file starting at
      byte offset 512 (after the header) and parses key-value pairs to
      populate object attributes. Automatically creates an MTime object
      for the initial start time using GPS time correction.

      :param fn: Path to Z3D file to read. Overrides instance fn if provided.
                 Can be string path or pathlib.Path object.
      :type fn: str | pathlib.Path, optional
      :param fid: Open file object for reading Z3D file. Overrides instance fid if provided.
                  Must be opened in binary mode ('rb').
      :type fid: BinaryIO, optional

      :raises UnicodeDecodeError: If schedule metadata cannot be decoded as UTF-8 text.
      :raises IndexError: If schedule lines don't match expected "Schedule.key = value" format.
      :raises ValueError: If Date/Time values cannot be parsed into valid MTime object.

      .. rubric:: Notes

      The method performs the following steps:
      1. Determines file source (fn parameter, fid parameter, or instance attributes)
      2. Seeks to byte offset 512 (after Z3D header)
      3. Reads 512 bytes of schedule metadata
      4. Splits metadata into lines and parses key-value pairs
      5. Sets object attributes for each parsed schedule entry
      6. Creates MTime object from Date and Time with GPS correction

      Schedule entries must follow the format "Schedule.key = value".
      The "Schedule." prefix is removed and "/" characters in keys are stripped
      (e.g., "S/R" becomes "SR").

      If both Date and Time are present, they are combined into an MTime object
      with GPS time correction applied automatically.

      .. rubric:: Examples

      Read from file path:

      >>> schedule = Z3DSchedule()
      >>> schedule.read_schedule(fn="recording.z3d")
      >>> print(f"Sampling rate: {schedule.SR}")

      Read from open file object:

      >>> with open("recording.z3d", "rb") as fid:
      ...     schedule = Z3DSchedule()
      ...     schedule.read_schedule(fid=fid)
      ...     print(f"GPS sync: {schedule.Sync}")

      Read using instance attributes:

      >>> schedule = Z3DSchedule(fn="recording.z3d")
      >>> schedule.read_schedule()  # Uses instance fn
      >>> print(f"Start time: {schedule.initial_start}")



