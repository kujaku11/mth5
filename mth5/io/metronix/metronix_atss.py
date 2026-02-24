# -*- coding: utf-8 -*-
"""
ATSS (Audio Time Series System) file reader for Metronix data.

This module provides functionality to read and process Metronix ATSS binary
time series files and their associated JSON metadata files. ATSS files contain
double precision floating point time series data equivalent to numpy arrays
of type np.float64.

The ATSS format consists of two files:
- .atss file: Binary time series data (np.float64 values)
- .json file: Metadata in JSON format

This implementation is translated from:
https://github.com/bfrmtx/MTHotel/blob/main/python/include/atss_file.py

Classes
-------
ATSS : MetronixFileNameMetadata
    Main class for reading ATSS files and converting to ChannelTS objects.

Functions
---------
read_atss : function
    Convenience function to read ATSS file and return ChannelTS object.

Notes
-----
ATSS files store time series data as consecutive double precision floating
point numbers in binary format, making them efficient for large datasets.

Examples
--------
>>> from mth5.io.metronix.metronix_atss import ATSS, read_atss
>>>
>>> # Using the ATSS class directly
>>> atss = ATSS('data/station001.atss')
>>> data = atss.read_atss()
>>> channel_ts = atss.to_channel_ts()
>>>
>>> # Using the convenience function
>>> channel_ts = read_atss('data/station001.atss')

Author
------
jpeacock

Created
-------
Tue Nov 26 15:54:12 2024
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from mt_metadata.timeseries import Run, Station, Survey
from mt_metadata.timeseries.auxiliary import Auxiliary
from mt_metadata.timeseries.electric import Electric
from mt_metadata.timeseries.filters import ChannelResponse
from mt_metadata.timeseries.magnetic import Magnetic

from mth5.io.metronix import MetronixChannelJSON, MetronixFileNameMetadata
from mth5.timeseries import ChannelTS


# =============================================================================


class ATSS(MetronixFileNameMetadata):
    """
    ATSS (Audio Time Series System) file reader for Metronix data.

    Handles reading and processing of Metronix ATSS binary time series files
    and their associated JSON metadata files. ATSS files contain double precision
    floating point time series data equivalent to numpy arrays of type np.float64.

    Parameters
    ----------
    fn : str or Path, optional
        Path to the ATSS file. If provided, metadata will be automatically
        loaded if the corresponding JSON file exists.
    **kwargs
        Additional keyword arguments passed to parent class.

    Attributes
    ----------
    header : MetronixChannelJSON
        Metadata handler for the associated JSON file.

    Notes
    -----
    ATSS files come in pairs:
    - .atss file: Binary time series data (np.float64)
    - .json file: Metadata in JSON format

    Examples
    --------
    >>> atss = ATSS('data/station001_run001_ch001.atss')
    >>> data = atss.read_atss()
    >>> channel_ts = atss.to_channel_ts()
    """

    def __init__(self, fn: str | Path | None = None, **kwargs: Any) -> None:
        super().__init__(fn=fn, **kwargs)

        self.header = MetronixChannelJSON()
        if self.has_metadata_file():
            self.header.read(self.metadata_fn)

    @property
    def metadata_fn(self) -> Path | None:
        """
        Path to the metadata JSON file.

        Returns the path to the JSON metadata file that corresponds to this
        ATSS file. The JSON file has the same base name as the ATSS file
        but with a .json extension.

        Returns
        -------
        Path or None
            Path to the JSON metadata file, or None if no ATSS file is set.

        Examples
        --------
        >>> atss = ATSS('data/station001.atss')
        >>> atss.metadata_fn
        PosixPath('data/station001.json')
        """
        if self.fn is not None:
            return self.fn.parent.joinpath(f"{self.fn.stem}.json")

    def has_metadata_file(self) -> bool:
        """
        Check if metadata JSON file exists.

        Returns
        -------
        bool
            True if the metadata JSON file exists, False otherwise.

        Examples
        --------
        >>> atss = ATSS('data/station001.atss')
        >>> atss.has_metadata_file()
        True
        """
        if self.fn is not None:
            return self.metadata_fn.exists()
        return False

    def read_atss(
        self, fn: str | Path | None = None, start: int = 0, stop: int = 0
    ) -> np.ndarray:
        """
        Read binary ATSS time series data.

        Reads double precision floating point time series data from the ATSS
        binary file. Data is stored as consecutive np.float64 values.

        Parameters
        ----------
        fn : str or Path, optional
            Path to ATSS file. If None, uses the current file path.
        start : int, default 0
            Starting sample index (0-based).
        stop : int, default 0
            Ending sample index. If 0, reads to end of file.

        Returns
        -------
        np.ndarray
            Time series data as 1D array of np.float64 values.

        Raises
        ------
        ValueError
            If stop index exceeds the number of samples in the file.

        Examples
        --------
        >>> atss = ATSS('data/station001.atss')
        >>> data = atss.read_atss()  # Read entire file
        >>> data_slice = atss.read_atss(start=1000, stop=2000)  # Read subset
        """
        if fn is not None:
            self.fn = fn

        if stop > self.n_samples:
            raise ValueError(f"stop {stop} > samples {self.n_samples}")
        with open(self.fn, "rb") as fid:
            # Read the binary data
            fid.seek(start * 8)
            # read in full file
            if stop == 0:
                data_bytes = fid.read()  # complete file
            else:
                data_bytes = fid.read(stop * 8)
            # Convert the data to a numpy array
            data_array = np.frombuffer(data_bytes, dtype=np.float64)

        return data_array

    def write_atss(self, data_array: np.ndarray, filename: str | Path) -> None:
        """
        Write time series data to ATSS binary file.

        Writes numpy array data as double precision floating point values
        to a binary ATSS file.

        Parameters
        ----------
        data_array : np.ndarray
            Time series data to write. Will be converted to np.float64.
        filename : str or Path
            Output file path for the ATSS binary file.

        Examples
        --------
        >>> import numpy as np
        >>> atss = ATSS()
        >>> data = np.random.randn(10000)
        >>> atss.write_atss(data, 'output.atss')
        """
        with open(filename, "wb") as fid:
            data_bytes = data_array.tobytes()
            fid.write(data_bytes)

    @property
    def channel_metadata(self) -> Electric | Magnetic | Auxiliary:
        """
        Channel metadata from the JSON header file.

        Returns
        -------
        Electric or Magnetic or Auxiliary
            Channel metadata object based on the channel type.
        """
        return self.header.get_channel_metadata()

    @property
    def channel_response(self) -> ChannelResponse:
        """
        Channel response information from the JSON header file.

        Returns
        -------
        ChannelResponse
            Channel response/calibration information.
        """
        return self.header.get_channel_response()

    @property
    def channel_type(self) -> str:
        """
        Determine channel type from component name.

        Channel type is determined from the component identifier in the filename:
        - Components starting with 'e': electric
        - Components starting with 'h': magnetic
        - All others: auxiliary

        Returns
        -------
        str
            Channel type: 'electric', 'magnetic', or 'auxiliary'.
        """
        if self.fn_exists:
            if self.component.startswith("e"):
                return "electric"
            elif self.component.startswith("h"):
                return "magnetic"
            else:
                return "auxiliary"

    @property
    def run_id(self) -> str | None:
        """
        Extract run ID from file path.

        Expects file path structure: .../station/run/timeseries.atss
        The run ID is extracted from the parent directory name.

        Returns
        -------
        str or None
            Run identifier, or None if file doesn't exist.
        """
        if self.fn.exists:
            return self.fn.parent.name

    @property
    def station_id(self) -> str | None:
        """
        Extract station ID from file path.

        Expects file path structure: .../station/run/timeseries.atss
        The station ID is extracted from the grandparent directory name.

        Returns
        -------
        str or None
            Station identifier, or None if file doesn't exist.
        """
        if self.fn.exists:
            return self.fn.parent.parent.name

    @property
    def survey_id(self) -> str | None:
        """
        Extract survey ID from file path.

        Expects file path structure: .../survey/stations/station/run/timeseries.atss
        The survey ID is extracted from the great-great-grandparent directory name.

        Returns
        -------
        str or None
            Survey identifier, or None if file doesn't exist.
        """
        if self.fn.exists:
            return self.fn.parent.parent.parent.parent.name

    @property
    def run_metadata(self) -> Run:
        """
        Generate run-level metadata.

        Creates a Run metadata object populated with information from the
        ATSS file and its associated JSON metadata.

        Returns
        -------
        Run
            Run metadata object with data logger info, sample rate,
            and channel metadata.
        """
        run = Run(id=self.run_id)
        run.data_logger.id = self.system_number
        run.data_logger.manufacturer = "Metronix Geophysics"
        run.data_logger.model = self.system_name
        run.sample_rate = self.sample_rate
        run.add_channel(self.channel_metadata)
        run.update_time_period()
        return run

    @property
    def station_metadata(self) -> Station:
        """
        Generate station-level metadata.

        Creates a Station metadata object populated with location information
        from the JSON metadata and run information.

        Returns
        -------
        Station
            Station metadata object with location coordinates and run metadata.
        """
        station = Station(id=self.station_id)
        station.location.latitude = self.header.metadata.latitude
        station.location.longitude = self.header.metadata.longitude
        station.location.elevation = self.header.metadata.elevation
        station.add_run(self.run_metadata)
        station.update_time_period()
        return station

    @property
    def survey_metadata(self) -> Survey:
        """
        Generate survey-level metadata.

        Creates a Survey metadata object that includes station metadata
        and overall time period information.

        Returns
        -------
        Survey
            Survey metadata object containing station information.
        """
        survey = Survey(id=self.survey_id)
        survey.add_station(self.station_metadata)
        survey.update_time_period()
        return survey

    def to_channel_ts(self, fn: str | Path | None = None) -> ChannelTS:
        """
        Create a ChannelTS object from ATSS data.

        Converts the ATSS time series data and metadata into a ChannelTS
        object suitable for use with MTH5 workflows.

        Parameters
        ----------
        fn : str or Path, optional
            Path to ATSS file. If None, uses current file path.

        Returns
        -------
        ChannelTS
            Time series object with data, metadata, and response information.

        Warnings
        --------
        Can be slow due to pandas datetime index creation for large datasets.
        A warning is logged if the metadata JSON file is missing.

        Examples
        --------
        >>> atss = ATSS('data/station001.atss')
        >>> channel_ts = atss.to_channel_ts()
        >>> print(channel_ts.sample_rate)
        1024.0
        """
        if not self.has_metadata_file():
            logger.warning(
                f"Could not find Metronix metadata JSON file for {self.fn.name}."
            )

        return ChannelTS(
            channel_type=self.channel_type,
            data=self.read_atss(),
            channel_metadata=self.channel_metadata,
            channel_response=self.channel_response,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            survey_metadata=self.survey_metadata,
        )


def read_atss(
    fn: str | Path,
    calibration_fn: str | Path | None = None,
    logger_file_handler: Any = None,
) -> ChannelTS:
    """
    Generic tool to read ATSS file and return ChannelTS object.

    Convenience function that creates an ATSS object and converts it
    to a ChannelTS in a single call.

    Parameters
    ----------
    fn : str or Path
        Path to the ATSS file to read.
    calibration_fn : str or Path, optional
        Path to calibration file (currently unused).
    logger_file_handler : Any, optional
        Logger file handler (currently unused).

    Returns
    -------
    ChannelTS
        Time series object with data and metadata from the ATSS file.

    Examples
    --------
    >>> channel_ts = read_atss('data/station001.atss')
    >>> print(f"Loaded {len(channel_ts.ts)} samples")
    """
    atss_obj = ATSS(fn)
    return atss_obj.to_channel_ts()


# ##################################################################################################################


# def stop_date_time(file_name):
#     nsamples = samples(file_name)
#     # get the sample_rate from the file name
#     channel = read_header(file_name)
#     # get the sample rate, the read_header function returns ensures that the sample rate is in Hz
#     sample_rate = channel["sample_rate"]
#     # get the start date time ISO 8601 like "1970-01-01T00:00:00.0"
#     start_date_time = channel["datetime"]
#     # calculate the stop date time
#     stop_date_time = np.datetime64(start_date_time) + np.timedelta64(
#         int(nsamples / sample_rate), "s"
#     )
#     return stop_date_time


# def duration(file_name):
#     # get the start date time ISO 8601 like "1970-01-01T00:00:00.0"
#     start_date_time = read_header(file_name)["datetime"]
#     # get the stop date time ISO 8601 like "1970-01-01T00:00:00.0"
#     stop_date_time = stop_date_time(file_name)
#     # calculate the duration
#     duration = stop_date_time - np.datetime64(start_date_time)
#     # return the duration in HH:MM:SS
#     return str(duration)


# # check if atss and json exist, and return the samples
# def exits_both(file_name):
#     sfile_name = atss_basename(file_name) + ".json"
#     # if not exist, terminate with FileNotFoundError
#     if not os.path.exists(sfile_name):
#         raise FileNotFoundError(f"File {sfile_name} not found")
#     #
#     sfile_name = atss_basename(file_name) + ".atss"
#     # if not exist, terminate with FileNotFoundError
#     if not os.path.exists(sfile_name):
#         raise FileNotFoundError(f"File {sfile_name} not found")
#     # if both exist, return the amount samples
#     return samples(file_name)


# def cal_mfs_06e(spc, file_name, wl):
#     # the calibration data for the MFS-06e sensor
#     # the spc is the complex spectrum, calculated by the fft "backward" function
#     # file_name is the file name of the channel, we take the sample rate from the header
#     #
#     # get the channel from the file
#     if not os.path.exists(file_name + ".json"):
#         raise FileNotFoundError(f"File {file_name}.json not found")
#     channel = read_header(file_name)
#     fs = channel["sample_rate"]
#     chopper = channel["sensor_calibration"]["chopper"]
#     if chopper == 1:
#         # calculate the frequency for each bin
#         for i, x in enumerate(spc):
#             if i == 0:
#                 continue
#             f = i * fs / wl
#             p1 = complex(0.0, (f / 4.0))
#             p2 = complex(0.0, (f / 8192.0))
#             p4 = complex(0.0, (f / 28300.0))
#             trf = 800.0 * (
#                 (p1 / (1.0 + p1)) * (1.0 / (1.0 + p2)) * (1.0 / (1.0 + p4))
#             )
#             spc[i] = spc[i] / trf
#     else:
#         # calculate the frequency for each bin
#         for i in enumerate(spc):
#             if i == 0:
#                 continue
#             f = i * fs / wl
#             p1 = complex(0.0, (f / 4.0))
#             p2 = complex(0.0, (f / 8192.0))
#             p3 = complex(0.0, (f / 0.720))
#             p4 = complex(0.0, (f / 28300.0))
#             trf = 800.0 * (
#                 (p1 / (1.0 + p1))
#                 * (1.0 / (1.0 + p2))
#                 * (p3 / (1.0 + p3))
#                 * (1.0 / (1.0 + p4))
#             )
#             spc[i] = spc[i] / trf
