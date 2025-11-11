# -*- coding: utf-8 -*-
"""
Z3D Schedule File Parser

This module provides functionality for parsing schedule information from Z3D files.
The Z3DSchedule class extracts and processes schedule metadata stored at offset 512
in Z3D files, providing access to various recording parameters and timing information.

Created on Wed Aug 24 11:24:57 2022
@author: jpeacock
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from loguru import logger

# =============================================================================
# Imports
# =============================================================================
from mt_metadata.common.mttime import MTime


# =============================================================================
class Z3DSchedule:
    """
    Parser for Z3D file schedule information and metadata.

    Reads schedule information from Z3D files and creates object attributes
    for each metadata entry. Schedule data is stored at byte offset 512 in
    Z3D files and contains recording parameters, timing information, and
    instrument configuration settings.

    The class preserves the original capitalization from the Z3D file format
    and provides automatic parsing of key-value pairs in the schedule section.

    Parameters
    ----------
    fn : Union[str, Path], optional
        Full path to Z3D file to read schedule information from.
        Can be string path or pathlib.Path object.
    fid : BinaryIO, optional
        Open file object for reading Z3D file in binary mode.
        Example: open('file.z3d', 'rb')
    **kwargs : Any
        Additional keyword arguments to set as object attributes.

    Attributes
    ----------
    AutoGain : str or None
        Auto gain setting for the recording channel ['Y' or 'N'].
    Comment : str or None
        User comments or notes for the schedule configuration.
    Date : str or None
        Date when the schedule action was started in YYYY-MM-DD format.
    Duty : str or None
        Duty cycle percentage of the transmitter (0-100).
    FFTStacks : str or None
        Number of FFT stacks used by the transmitter.
    Filename : str or None
        Original filename that the ZEN instrument assigns to the recording.
    Gain : str or None
        Gain setting for the recording channel (e.g., '1.0000').
    Log : str or None
        Data logging enabled flag ['Y' or 'N'].
    NewFile : str or None
        Create new file for recording flag ['Y' or 'N'].
    Period : str or None
        Base period setting for the transmitter in seconds.
    RadioOn : str or None
        Radio communication enabled flag ['Y', 'N', or 'X'].
    SR : str or None
        Sampling rate in Hz (originally 'S/R' in file, converted to 'SR').
    SamplesPerAcq : str or None
        Number of samples per acquisition for transmitter mode.
    Sleep : str or None
        Sleep mode enabled flag ['Y' or 'N'].
    Sync : str or None
        GPS synchronization enabled flag ['Y' or 'N'].
    Time : str or None
        Time when the schedule action started in HH:MM:SS format (GPS time).
    initial_start : MTime
        Parsed start time as MTime object with GPS time flag enabled.
        Combines Date and Time attributes for timestamp calculation.
    fn : Union[str, Path] or None
        Path to the Z3D file being processed.
    fid : BinaryIO or None
        File object for reading the Z3D file.
    meta_string : bytes or None
        Raw schedule metadata string read from the file.
    _header_len : int
        Length of Z3D file header in bytes (512).
    _schedule_metadata_len : int
        Length of schedule metadata section in bytes (512).
    logger : Logger
        Loguru logger instance for debugging and status messages.

    Notes
    -----
    The Z3D file format stores schedule information at a fixed offset of 512 bytes
    from the beginning of the file. The schedule section is also 512 bytes long
    and contains key-value pairs in the format "Schedule.key = value".

    All schedule values are stored as strings to preserve the original format
    from the Z3D file. Boolean-like values use 'Y'/'N' convention.

    The initial_start attribute automatically converts Date and Time into a
    GPS-corrected MTime object for accurate timestamp handling.

    Examples
    --------
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
    """

    def __init__(
        self,
        fn: Optional[Union[str, Path]] = None,
        fid: Optional[BinaryIO] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Z3DSchedule parser.

        Parameters
        ----------
        fn : Union[str, Path], optional
            Path to Z3D file. Can be string or pathlib.Path object.
        fid : BinaryIO, optional
            Open file object for reading Z3D file in binary mode.
        **kwargs : Any
            Additional attributes to set on the instance.
        """
        self.logger = logger
        self.fn: Optional[Union[str, Path]] = fn
        self.fid: Optional[BinaryIO] = fid
        self.meta_string: Optional[bytes] = None

        # Z3D file format constants
        self._schedule_metadata_len: int = 512
        self._header_len: int = 512

        # Schedule metadata attributes (all stored as strings from Z3D format)
        self.AutoGain: Optional[str] = None
        self.Comment: Optional[str] = None
        self.Date: Optional[str] = None
        self.Duty: Optional[str] = None
        self.FFTStacks: Optional[str] = None
        self.Filename: Optional[str] = None
        self.Gain: Optional[str] = None
        self.Log: Optional[str] = None
        self.NewFile: Optional[str] = None
        self.Period: Optional[str] = None
        self.RadioOn: Optional[str] = None
        self.SR: Optional[str] = None  # Sampling Rate (S/R becomes SR)
        self.SamplesPerAcq: Optional[str] = None
        self.Sleep: Optional[str] = None
        self.Sync: Optional[str] = None
        self.Time: Optional[str] = None

        # Parsed start time with GPS correction
        self.initial_start: MTime = MTime(time_stamp=None)

        # Set any additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def read_schedule(
        self, fn: Optional[Union[str, Path]] = None, fid: Optional[BinaryIO] = None
    ) -> None:
        """
        Read and parse schedule metadata from Z3D file.

        Reads the schedule information section from a Z3D file starting at
        byte offset 512 (after the header) and parses key-value pairs to
        populate object attributes. Automatically creates an MTime object
        for the initial start time using GPS time correction.

        Parameters
        ----------
        fn : Union[str, Path], optional
            Path to Z3D file to read. Overrides instance fn if provided.
            Can be string path or pathlib.Path object.
        fid : BinaryIO, optional
            Open file object for reading Z3D file. Overrides instance fid if provided.
            Must be opened in binary mode ('rb').

        Raises
        ------
        UnicodeDecodeError
            If schedule metadata cannot be decoded as UTF-8 text.
        IndexError
            If schedule lines don't match expected "Schedule.key = value" format.
        ValueError
            If Date/Time values cannot be parsed into valid MTime object.

        Notes
        -----
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

        Examples
        --------
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
        """
        # Update file parameters if provided
        if fn is not None:
            self.fn = fn
        if fid is not None:
            self.fid = fid

        # Validate file source is available
        if self.fn is None and self.fid is None:
            self.logger.warning("No Z3D file to read.")
            return

        # Read schedule metadata from file
        if self.fn is None:
            # Use existing file object
            if self.fid is not None:
                self.fid.seek(self._header_len)
                self.meta_string = self.fid.read(self._schedule_metadata_len)
        elif self.fn is not None:
            # Open file and read, or use existing file object
            if self.fid is None:
                self.fid = open(self.fn, "rb")
                self.fid.seek(self._header_len)
                self.meta_string = self.fid.read(self._schedule_metadata_len)
            else:
                self.fid.seek(self._header_len)
                self.meta_string = self.fid.read(self._schedule_metadata_len)

        # Parse schedule metadata lines
        meta_list = self.meta_string.split(b"\n")
        for m_str in meta_list:
            try:
                m_str_decoded = m_str.decode()
                if m_str_decoded.find("=") > 0:
                    m_list = m_str_decoded.split("=")
                    # Extract key after "Schedule." prefix
                    m_key = m_list[0].split(".")[1].strip()
                    # Clean key name (remove slashes)
                    m_key = m_key.replace("/", "")
                    # Extract and clean value
                    m_value = m_list[1].strip()
                    # Set as object attribute
                    setattr(self, m_key, m_value)
            except (UnicodeDecodeError, IndexError) as e:
                # Skip malformed lines but log for debugging
                self.logger.debug(f"Skipped malformed schedule line: {m_str!r} - {e}")
                continue

        # Create initial start time from Date and Time with GPS correction
        try:
            if self.Date is not None and self.Time is not None:
                timestamp_str = f"{self.Date}T{self.Time}"
                self.initial_start = MTime(time_stamp=timestamp_str, gps_time=True)
        except Exception as e:
            self.logger.warning(f"Could not parse initial start time: {e}")
            # Keep default MTime object if parsing fails
