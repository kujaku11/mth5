"""
=======================================================================
original comments from MATLAB script:

read_tsn - reads a (binary) TS file of the legacy Phoenix MTU-5A instrument
(TS2, TS3, TS4, TS5) and the even older V5-2000 system (TSL, TSH), and
output the "ts" array and "tag" metadata dictionary.

=======================================================================
Parameters:
    fpath: path to the TS file
    fname: name of the TS file (including extensions)

Returns:
    ts:    output array of the TS data
    tag:   output dict of the TSn metadata

=======================================================================
definition of the TS tag (or what I guessed after reading the user manual
and fiddling with their files)
0-7   UTC time of first scan in the record.
8-9   instrument serial number (16-bit integer)
10-11 number of scans in the record (16-bit integer)
12    number of channels per scan
13    tag length (TSn) or tag length code (TSH, TSL)
14    status code
15    bit-wise saturation flags (please note that the older TSH/L tag
      ends here )
16    reserved for future indication of different tag and/or sample
      formats
17    sample length in bytes
18-19 sample rate (in units defined by byte 20)
20    units of sample rate
21    clock status
22-25 clock error in seconds
26-32 reserved; must be 0

=======================================================================
notes on the TS format of TSn files:
The binary TS file consists of several data blocks, each contains a data
tag and a number of records in it.
Each time record consists of three bytes (24 bit), let's name them byte1,
byte2, and byte3:
the ts record (int) should be (+/-) (byte3*65536 + byte2*256 + byte1)

Hao
2012.07.04
Beijing
=======================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
from loguru import logger
from mt_metadata.common import MTime

from mth5.timeseries import ChannelTS, RunTS

from .mtu_table import MTUTable


class MTUTSN:
    """
    Reader for legacy Phoenix MTU-5A instrument time series binary files.

    Reads time series data from Phoenix MTU-5A (.TS2, .TS3, .TS4, .TS5) and
    V5-2000 system (.TSL, .TSH) binary files. The data consists of 24-bit
    signed integers organized in data blocks with headers.

    Parameters
    ----------
    file_path : str or Path or None, optional
        Path to the TSN file to read. If None, the reader is created without
        loading data. Default is None.

    Attributes
    ----------
    file_path : Path or None
        Path to the currently loaded TSN file.
    ts : ndarray or None
        Time series data array with shape (n_channels, n_samples).
    tag : dict
        Metadata dictionary containing file information.

    Examples
    --------
    Read a TS3 file:

    >>> from pathlib import Path
    >>> reader = MTUTSN('data/1690C16C.TS3')
    >>> print(reader.ts.shape)
    (3, 86400)
    >>> print(reader.tag['sample_rate'])
    24

    Create reader without loading data:

    >>> reader = MTUTSN()
    >>> reader.read('data/1690C16C.TS3')

    Access metadata:

    >>> reader = MTUTSN('data/1690C16C.TS4')
    >>> reader.read()
    >>> print(f"Channels: {reader.tag['n_ch']}")
    Channels: 4
    >>> print(f"Blocks: {reader.tag['n_block']}")
    Blocks: 48
    """

    def __init__(self, file_path: str | Path | None = None, **kwargs) -> None:
        self._p16 = 2**16
        self._p8 = 2**8
        self._accepted_extensions = ["TS2", "TS3", "TS4", "TS5", "TSL", "TSH"]
        self._file_path = None

        self.file_path = file_path
        self.ts = None
        self.ts_metadata = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def file_path(self) -> Path | None:
        """Get the TSN file path."""
        return self._file_path

    @file_path.setter
    def file_path(self, value: str | Path | None) -> None:
        """
        Set the TSN file path with validation.

        Parameters
        ----------
        value : str or Path or None
            Path to the TSN file. Must exist and have a valid extension
            (.TS2, .TS3, .TS4, .TS5, .TSL, .TSH).

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file extension is not recognized.

        Examples
        --------
        >>> reader = MTUTSN()
        >>> reader.file_path = 'data/1690C16C.TS3'
        >>> print(reader.file_path.name)
        1690C16C.TS3
        """
        if value is not None:
            self._file_path = Path(value)
            if not self._file_path.exists():
                msg = f"File path {self._file_path} does not exist."
                logger.error(msg)
                raise FileNotFoundError(msg)
            if (
                self._file_path.suffix.upper().lstrip(".")
                not in self._accepted_extensions
            ):
                msg = (
                    f"File extension {self._file_path.suffix} is not a recognized "
                    f"TSN format. Accepted extensions: {self._accepted_extensions}"
                )
                logger.error(msg)
                raise ValueError(msg)
        else:
            self._file_path = None

    def get_sign24(self, x: np.ndarray | list | int) -> np.ndarray:
        """
        Convert unsigned 24-bit integers to signed integers.

        Converts unsigned 24-bit values (0 to 16777215) to their signed
        equivalents (-8388608 to 8388607) by applying two's complement.

        Parameters
        ----------
        x : ndarray or list or int
            Unsigned 24-bit integer value(s) to convert.

        Returns
        -------
        ndarray
            Signed 24-bit integer value(s) as int32 array.

        Examples
        --------
        Convert a single positive value:

        >>> reader = MTUTSN()
        >>> reader.get_sign24(100)
        array([100], dtype=int32)

        Convert a single negative value (unsigned representation):

        >>> reader.get_sign24(16777215)  # -1 in 24-bit signed
        array([-1], dtype=int32)

        Convert an array:

        >>> values = np.array([0, 8388607, 8388608, 16777215])
        >>> reader.get_sign24(values)
        array([       0,  8388607, -8388608,       -1], dtype=int32)
        """
        x = np.array(x, dtype=np.int32)
        x[x > 2**23 - 1] = x[x > 2**23 - 1] - 2**24
        return x

    def _read_header(
        self, ts_fid: BinaryIO
    ) -> tuple[MTime, int, int, int, int, str, int]:
        """
        Read and parse the 32-byte header from a TSN file.

        Extracts timestamp, instrument serial number, number of scans,
        channel count, tag length, instrument type, and sample rate from
        the binary file header.

        Parameters
        ----------
        ts_fid : BinaryIO
            Open binary file handle positioned at the start of the header.

        Returns
        -------
        start_time : MTime
            UTC timestamp of the first scan in the file.
        box_num : int
            Instrument serial number (16-bit integer).
        n_scan : int
            Number of scans per data block (16-bit integer).
        n_ch : int
            Number of channels per scan (3, 4, 5, or 6).
        tag_length : int
            Length of the tag in bytes (32 for MTU-5, 16 for V5-2000).
        ts_type : str
            Instrument type: 'MTU-5' or 'V5-2000'.
        sample_rate : int
            Sampling frequency in Hz (0 for V5-2000 files).

        Examples
        --------
        >>> with open('data/1690C16C.TS3', 'rb') as f:
        ...     reader = MTUTSN()
        ...     start, box, scan, ch, tag_len, ts_type, sr = reader._read_header(f)
        ...     print(f"Type: {ts_type}, Rate: {sr} Hz, Channels: {ch}")
        Type: MTU-5, Rate: 24 Hz, Channels: 3
        """
        # Starting time
        s = ts_fid.read(1)[0]  # Starting second
        m = ts_fid.read(1)[0]  # Starting minute
        h = ts_fid.read(1)[0]  # Starting hour
        d = ts_fid.read(1)[0]  # Starting day
        l = ts_fid.read(1)[0]  # Starting month
        y = ts_fid.read(1)[0]  # Starting year
        ts_fid.read(1)  # skip the Starting weekday
        c = ts_fid.read(1)[0]  # Starting century(-1)
        start = MTime(time_stamp=f"{(c*100)+y}-{l:02d}-{d:02d}T{h:02d}:{m:02d}:{s:02d}")
        logger.info(f"Start time is: {start.isoformat()} UTC")

        # Box series number (16-bit integer)
        box_num = int.from_bytes(ts_fid.read(2), byteorder="little", signed=False)
        # Number of scans in a data block (16-bit integer)
        n_scan = int.from_bytes(ts_fid.read(2), byteorder="little", signed=False)
        # Number of channels in a record
        n_ch = ts_fid.read(1)[0]
        # Length of the tag
        tag_length = ts_fid.read(1)[0]

        if tag_length != 32:
            ts_type = "V5-2000"
            sample_rate = 0
        else:
            ts_type = "MTU-5"
            ts_fid.seek(4, 1)  # skip to sampling frequency
            # Sampling frequency (16-bit integer, little endian)
            sample_rate = int.from_bytes(
                ts_fid.read(2), byteorder="little", signed=False
            )
            ts_fid.seek(12, 1)  # skip some (unknown) head info...
            logger.info(f"Sampling frequency is {sample_rate} Hz")
            logger.info(f"Number of records is {n_scan} in each data block")

        logger.info(f"TS type is: {ts_type}")
        return start, box_num, n_scan, n_ch, tag_length, ts_type, sample_rate

    def _extract_channel_data(
        self, data: np.ndarray, byte_indices: list[int]
    ) -> np.ndarray:
        """
        Extract and convert 24-bit signed channel data from raw bytes.

        Combines three consecutive bytes (low, middle, high) per sample to
        reconstruct 24-bit signed integer values for a single channel.

        Parameters
        ----------
        data : ndarray
            Raw byte data array with shape (n_ch*3, n_scan), where each row
            represents one byte position and each column is one time sample.
        byte_indices : list of int
            Three byte indices [low_byte, mid_byte, high_byte] indicating which
            rows of the data array contain the 24-bit value components.

        Returns
        -------
        ndarray
            Signed 24-bit integer values for the channel with shape (n_scan,).

        Examples
        --------
        Extract channel 0 from 3-channel data:

        >>> data = np.random.randint(0, 256, size=(9, 1000), dtype=np.int32)
        >>> reader = MTUTSN()
        >>> ch0_data = reader._extract_channel_data(data, [0, 1, 2])
        >>> print(ch0_data.shape)
        (1000,)

        Extract channel 1 from 4-channel data:

        >>> data = np.random.randint(0, 256, size=(12, 500), dtype=np.int32)
        >>> ch1_data = reader._extract_channel_data(data, [3, 4, 5])
        >>> print(ch1_data.shape)
        (500,)
        """
        return self.get_sign24(
            data[byte_indices[2], :] * self._p16
            + data[byte_indices[1], :] * self._p8
            + data[byte_indices[0], :]
        )

    def _process_data_block(
        self,
        ts_fid: BinaryIO,
        n_ch: int,
        n_scan: int,
        ts: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> bool:
        """
        Read and process a single data block from the TSN file.

        Reads one data block consisting of a 32-byte header followed by
        n_scan*n_ch*3 bytes of 24-bit channel data. Extracts and converts
        the data for all channels and populates the specified slice of the
        time series array.

        Parameters
        ----------
        ts_fid : BinaryIO
            Open binary file handle positioned at the start of a data block.
        n_ch : int
            Number of channels (3, 4, 5, or 6).
        n_scan : int
            Number of time samples in this block.
        ts : ndarray
            Pre-allocated time series array to populate with shape
            (n_ch, total_samples).
        start_idx : int
            Starting index in the second dimension of ts for this block's data.
        end_idx : int
            Ending index (exclusive) in the second dimension of ts.

        Returns
        -------
        bool
            True if data was read and processed successfully, False if end of
            file reached or unsupported channel count.

        Examples
        --------
        Process blocks from a file:

        >>> with open('data/1690C16C.TS3', 'rb') as f:
        ...     reader = MTUTSN()
        ...     # ... read header first ...
        ...     ts = np.zeros((3, 3600))
        ...     success = reader._process_data_block(f, 3, 1200, ts, 0, 1200)
        ...     print(f"Block processed: {success}")
        Block processed: True
        """
        ts_fid.seek(32, 1)  # skip the file tag
        data = np.frombuffer(ts_fid.read(n_ch * 3 * n_scan), dtype=np.uint8)

        if len(data) == 0:
            logger.warning("No data read in current block...")
            return False

        # Reshape data to [Nch*3, Nscan] (Fortran order to match MATLAB)
        # Convert to int32 to avoid overflow during multiplication
        data = data.reshape((n_scan, n_ch * 3)).T.astype(np.int32)

        # Channel byte index mapping (each channel uses 3 consecutive bytes)
        # Format: [low_byte, mid_byte, high_byte]
        channel_map = {
            3: {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
            4: {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [12, 13, 14]},
            5: {
                0: [0, 1, 2],
                1: [3, 4, 5],
                2: [6, 7, 8],
                3: [9, 10, 11],
                4: [12, 13, 14],
            },
            6: {
                0: [0, 1, 2],
                1: [3, 4, 5],
                2: [6, 7, 8],
                3: [9, 10, 11],
                4: [12, 13, 14],
                5: [15, 16, 17],
            },
        }

        if n_ch in channel_map:
            for ch_idx, byte_indices in channel_map[n_ch].items():
                ts[ch_idx, start_idx:end_idx] = self._extract_channel_data(
                    data, byte_indices
                )
        else:
            logger.warning(f"Unsupported number of channels: {n_ch}")
            return False

        return True

    def read(self, file_path: str | Path | None = None) -> None:
        """
        Read and parse a Phoenix MTU time series binary file.

        Reads complete time series data from legacy Phoenix MTU-5A instrument
        files (.TS2, .TS3, .TS4, .TS5) or V5-2000 system files (.TSL, .TSH).
        Each file contains multiple data blocks with 24-bit signed integer
        samples organized by channel.

        Parameters
        ----------
        file_path : str or Path or None, optional
            Path to the TSN file to read. If None, uses the current file_path
            attribute. Default is None.

        Returns
        -------
        ts : ndarray
            Time series data array with shape (n_channels, total_samples).
            Data type is float64. Each row represents one channel, and each
            column is a time sample.
        tag : dict
            Metadata dictionary containing file information with keys:

            - 'box_number' (int): Instrument serial number
            - 'ts_type' (str): Instrument type ('MTU-5' or 'V5-2000')
            - 'sample_rate' (int): Sampling frequency in Hz
            - 'n_ch' (int): Number of channels
            - 'n_scan' (int): Number of scans per data block
            - 'start' (MTime): UTC timestamp of first sample
            - 'ts_length' (float): Duration of each block in seconds
            - 'n_block' (int): Total number of data blocks in file

        Raises
        ------
        EOFError
            If the file is empty or cannot be read.
        ValueError
            If the file has an unsupported extension or channel count.
        FileNotFoundError
            If the specified file does not exist.

        Examples
        --------
        Read a 3-channel TS3 file:

        >>> reader = MTUTSN()
        >>> ts, tag = reader.read('data/1690C16C.TS3')
        >>> print(f"Shape: {ts.shape}")
        Shape: (3, 86400)
        >>> print(f"Sample rate: {tag['sample_rate']} Hz")
        Sample rate: 24 Hz
        >>> print(f"Duration: {ts.shape[1] / tag['sample_rate']:.1f} seconds")
        Duration: 3600.0 seconds

        Read a 4-channel TS4 file:

        >>> reader = MTUTSN('data/1690C16C.TS4')
        >>> print(f"Channels: {reader.tag['n_ch']}")
        Channels: 4
        >>> print(f"Start time: {reader.tag['start'].isoformat()}")
        Start time: 2016-07-16T00:00:00+00:00

        Read and process data:

        >>> ts, tag = MTUTSN().read('data/station.TS5')
        >>> # Calculate statistics for each channel
        >>> for i in range(tag['n_ch']):
        ...     print(f"Ch{i} mean: {ts[i].mean():.2f}, std: {ts[i].std():.2f}")
        Ch0 mean: 123.45, std: 456.78
        Ch1 mean: -234.56, std: 567.89
        ...
        """
        # try opening the ts data file
        if file_path is not None:
            self.file_path = file_path

        logger.info(f"Opening file: {self.file_path.name} ...")

        with open(self.file_path, "rb") as ts_fid:
            # Read the header to check if file is empty
            s_byte = ts_fid.read(1)  # Starting second
            if len(s_byte) == 0:
                raise EOFError("TSN file is empty or could not be read.")

            # Reset to beginning and read full header
            ts_fid.seek(0, 0)
            (
                start,
                box_num,
                n_scan,
                n_ch,
                tag_length,
                ts_type,
                sample_rate,
            ) = self._read_header(ts_fid)

            # Calculate file size and number of blocks
            ts_fid.seek(0, 2)
            file_size = ts_fid.tell()
            n_block = round(file_size / (n_scan * n_ch * 3 + 32))

            # Preallocate memory for time series data
            ts = np.zeros((n_ch, n_scan * n_block), dtype=np.float64)
            logger.info(f"Total {n_block} block(s) found in current file")

            # Reset to beginning of file and process each data block
            ts_fid.seek(0, 0)

            for iblock in range(n_block):
                start_idx = iblock * n_scan
                end_idx = (iblock + 1) * n_scan

                if not self._process_data_block(
                    ts_fid, n_ch, n_scan, ts, start_idx, end_idx
                ):
                    break

        logger.info("# finish reading time series...")

        # Build the tag dictionary
        tag = {
            "box_number": box_num,
            "ts_type": ts_type,
            "sample_rate": sample_rate,
            "n_ch": n_ch,
            "n_scan": n_scan,
            "start": start,
            "ts_length": n_scan / sample_rate if sample_rate > 0 else 0,
            "n_block": n_block,
        }

        self.ts = ts
        self.ts_metadata = tag

    def to_runts(
        self, table_filepath: str | Path | None = None, calibrate=True
    ) -> RunTS:
        """
        Create an MTUTable object from the TSN file and associated TBL file.

        Parameters
        ----------
        table_filepath : str or Path
            Path to the corresponding TBL file.

        Returns
        -------
        MTUTable
            An MTUTable object containing metadata from the TBL file.

        Examples
        --------
        >>> reader = MTUTSN('data/1690C16C.TS3')
        >>> mtu_table = reader.to_runts('data/1690C16C.TBL')
        >>> print(mtu_table.metadata)
        {...}
        """
        # Read data if not already loaded
        if self.ts is None or self.ts_metadata is None:
            self.read()

        ts = self.ts
        ts_metadata = self.ts_metadata

        if table_filepath is None and self.file_path is not None:
            table_filepath = self.file_path.with_suffix(".TBL")

        mtu_table = MTUTable(table_filepath)
        survey_metadata = mtu_table.survey_metadata  # to trigger warning if no data
        run_metadata = mtu_table.run_metadata.copy()
        run_metadata.sample_rate = ts_metadata["sample_rate"]
        # update run id to include sample rate
        run_metadata.id = f"sr{ts_metadata['sample_rate']}_001"
        run_ts = RunTS(
            survey_metadata=mtu_table.survey_metadata,
            station_metadata=survey_metadata.stations[0],
            run_metadata=run_metadata,
        )
        for comp, channel_number in mtu_table.channel_keys.items():
            channel_metadata = getattr(mtu_table, f"{comp}_metadata")
            channel_metadata.sample_rate = ts_metadata["sample_rate"]
            channel_metadata.start_time = ts_metadata["start"]

            # Channel numbers in TBL are 1-indexed, convert to 0-indexed for numpy
            channel_index = channel_number - 1

            if calibrate:
                if comp in ["ex", "ey"]:
                    scale_factor = (
                        mtu_table.ex_calibration
                        if comp == "ex"
                        else mtu_table.ey_calibration
                    )
                    channel_metadata.units = "mV/km"

                elif comp in ["hx", "hy", "hz"]:
                    scale_factor = mtu_table.magnetic_calibration
                    channel_metadata.units = "nT"

                logger.info(
                    f"Applying scale factor of {scale_factor} to channel {comp}"
                )
                ts[channel_index, :] = ts[channel_index, :] * scale_factor

            # Determine channel type
            channel_type = "electric" if comp[0] in ["e"] else "magnetic"

            ch_ts = ChannelTS(
                channel_type=channel_type,
                data=ts[channel_index, :],
                channel_metadata=channel_metadata,
            )
            run_ts.add_channel(ch_ts)

        return run_ts
