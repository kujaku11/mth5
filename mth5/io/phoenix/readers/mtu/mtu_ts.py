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

from pathlib import Path

import numpy as np
from loguru import logger
from mt_metadata.common import MTime


class MTUTSN:
    """
    A class to read the legacy Phoenix MTU-5A instrument time series binary files
    (.TSN) format.
    """

    def __init__(self, file_path: str | Path | None = None):
        self._p16 = 2**16
        self._p8 = 2**8
        self._accepted_extensions = ["TS2", "TS3", "TS4", "TS5", "TSL", "TSH"]
        self._file_path = None

        self.file_path = file_path

        if self.file_path is not None:
            self.ts, self.tag = self.read()
        else:
            self.ts, self.tag = None, {}

    @property
    def file_path(self) -> Path | None:
        """Get the TSN file path."""
        return self._file_path

    @file_path.setter
    def file_path(self, value: str | Path | None):
        """Set the TSN file path."""
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

    def get_sign24(self, x):
        """
        a simple function to calculate the sign for a 24 bit number
        I should have made it in-line
        """
        x = np.array(x, dtype=np.int32)
        x[x > 2**23 - 1] = x[x > 2**23 - 1] - 2**24
        return x

    def _read_header(self, ts_fid) -> tuple[MTime, int, int, int, int, str, int]:
        """
        Read the header information from the TSN file.

        Parameters
        ----------
        ts_fid : file object
            Open file handle positioned at the start of the file.

        Returns
        -------
        tuple
            (start_time, box_num, n_scan, n_ch, tag_length, ts_type, sample_rate)
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
        Extract a single channel's 24-bit signed data from the raw byte array.

        Parameters
        ----------
        data : np.ndarray
            Raw data array with shape (n_ch*3, n_scan).
        byte_indices : list[int]
            Three byte indices [low, mid, high] for the 24-bit value.

        Returns
        -------
        np.ndarray
            Signed 24-bit integer values for the channel.
        """
        return self.get_sign24(
            data[byte_indices[2], :] * self._p16
            + data[byte_indices[1], :] * self._p8
            + data[byte_indices[0], :]
        )

    def _process_data_block(
        self,
        ts_fid,
        n_ch: int,
        n_scan: int,
        ts: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> bool:
        """
        Process a single data block and populate the ts array.

        Parameters
        ----------
        ts_fid : file object
            Open file handle positioned at a data block.
        n_ch : int
            Number of channels.
        n_scan : int
            Number of scans per block.
        ts : np.ndarray
            Time series array to populate.
        start_idx : int
            Starting index in the ts array.
        end_idx : int
            Ending index in the ts array.

        Returns
        -------
        bool
            True if data was read successfully, False if no data.
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

    def read(
        self, file_path: str | Path | None = None
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        read_tsn - reads a (binary) TS file of the legacy Phoenix MTU-5A instrument
        (TS2, TS3, TS4, TS5) and the even older V5-2000 system (TSL, TSH), and
        output the "ts" array and "tag" metadata dictionary.

        Parameters:
            fpath: path to the TS file
            fname: name of the TS file (including extensions)

        Returns:
            ts:    output numpy array of the TS data
            tag:   output dict of the TSn metadata
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

        return ts, tag
