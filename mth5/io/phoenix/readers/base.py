"""Module to read and parse native Phoenix Geophysics data formats of the MTU-5C Family

This module implements Streamed readers for segmented-decimated continuus-decimated
and native sampling rate time series formats of the MTU-5C family.
"""

__author__ = "Jorge Torres-Solis"

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from .header import Header

from mth5.utils.mth5_logger import setup_logger

# =============================================================================


class TSReaderBase(Header):
    """

    Generic reader that all other readers will inherit

    """

    def __init__(
        self, path, num_files=1, header_length=128, report_hw_sat=False, **kwargs
    ):
        self._seq = None
        super().__init__(
            header_length=header_length, report_hw_sat=report_hw_sat, **kwargs
        )

        self.logger = setup_logger(f"{self.__class__}.{self.__class__.__name__}")
        self.base_path = path
        self.last_seq = self.seq + num_files
        self.stream = None
        # Open the file passed as the first file in the sequence to stream
        self._open_file(self.base_path)
        if self._recording_id is None:
            self.recording_id = self.base_path.stem.split("_")[1]
        if self._channel_id is None:
            self.channel_id = self.base_path.stem.split("_")[2]

    @property
    def base_path(self):
        """

        :return: full path of file
        :rtype: :class:`pathlib.Path`

        """
        return self._base_path

    @base_path.setter
    def base_path(self, value):
        """

        :param value: full path to file
        :type value: string or :class:`pathlib.Path`

        """

        self._base_path = Path(value)

    @property
    def base_dir(self):
        """

        :return: parent directory of file
        :rtype: :class:`pathlib.Path`

        """
        return self.base_path.parent

    @property
    def file_name(self):
        """

        :return: name of the file
        :rtype: string

        """
        return self.base_path.name

    @property
    def file_extension(self):
        """

        :return: file extension
        :rtype: string

        """
        return self.base_path.suffix

    @property
    def instrument_id(self):
        """

        :return: instrument ID
        :rtype: string

        """
        return self.base_path.stem.split("_")[0]

    @property
    def seq(self):
        """

        :return: sequence number of the file
        :rtype: int

        """
        if self._seq is None:
            return int(self.base_path.stem.split("_")[3], 16)
        return self._seq

    @seq.setter
    def seq(self, value):
        """

        :param value: sequence number
        :type value: integer


        """
        self._seq = int(value)

    @property
    def file_size(self):
        """

        :return: file size in bytes
        :rtype: integer

        """
        return self.base_path.stat().st_size

    @property
    def max_samples(self):
        """
        Max number of samples in a file which is:

        (total number of bytes - header length) / frame size * n samples per frame

        :return: max number of samples in a file
        :rtype: int

        """
        return int((self.file_size - self.header_length) / 64 * 20)

    @property
    def sequence_list(self):
        """
        get all the files in the sequence sorted by sequence number
        """
        return sorted(list(self.base_dir.glob(f"*{self.file_extension}")))

    def _open_file(self, filename):
        """
        open a given file in 'rb' mode

        :param filename: full path to file
        :type filename: :class:`pathlib.Path`
        :return: boolean if the file is now open [True] or not [False]
        :rtype: boolean

        """
        filename = Path(filename)

        if filename.exists():
            self.logger.debug(f"Opening {filename}")
            self.stream = open(filename, "rb")
            return True
        return False

    def open_next(self):
        """
        Open the next file in the sequence
        :return: [True] if next file is now open, [False] if it is not
        :rtype: boolean

        """
        if self.stream is not None:
            self.stream.close()
        self.seq += 1
        self.open_file_seq(self.seq)
        if self.seq < self.last_seq:
            new_path = self.sequence_list[self.seq - 1]
            return self._open_file(new_path)
        return False

    def open_file_seq(self, file_seq_num=None):
        """
        Open a file in the sequence given the sequence number
        :param file_seq_num: sequence number to open, defaults to None
        :type file_seq_num: integer, optional
        :return: [True] if next file is now open, [False] if it is not
        :rtype: boolean

        """
        if self.stream is not None:
            self.stream.close()
        if file_seq_num is not None:
            self.seq = file_seq_num
        new_path = self.sequence_list[self.seq - 1]
        return self._open_file(new_path)

    def close(self):
        """
        Close the file

        """
        if self.stream is not None:
            self.stream.close()
