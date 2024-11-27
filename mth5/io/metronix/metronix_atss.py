# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:54:12 2024

Translated from 
https://github.com/bfrmtx/MTHotel/blob/main/python/include/atss_file.py

the atss files are two files; one for the header.json and one for the data.atss
both have the same name, but different extensions
the header.json contains the metadata and the data.atss contains the time series data
data.atss is a binary file, containing double precision floating point numbers
that is equivalent to a numpy array of type np.float64

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import json
import copy
import numpy as np
from loguru import logger

from mth5.timeseries import ChannelTS
from mth5.io.metronix import MetronixFileNameMetadata, MetronixChannelJSON

# =============================================================================


class ATSS(MetronixFileNameMetadata):
    def __init__(self, fn=None, **kwargs):
        super().__init__(fn, **kwargs)

    @property
    def metadata_fn(self):
        """metadata JSON file, same name as the atss file with extension json"""
        if self.fn is not None:
            return self.fn.parent.joinpath(f"{self.fn.stem}.json")

    def has_metadata(self):
        """has metadata file (.json)"""
        if self.fn is not None:
            return self.metadata_fn.exists()
        return False

    def read_atss(self, fn=None, start=0, stop=0):
        """

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :param start: DESCRIPTION, defaults to 0
        :type start: TYPE, optional
        :param wl: DESCRIPTION, defaults to 0
        :type wl: TYPE, optional
        :raises ValueError: DESCRIPTION
        :return: DESCRIPTION
        :rtype: TYPE

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

    def write_atss(self, data_array, filename):
        with open(filename, "wb") as fid:
            data_bytes = data_array.tobytes()
            fid.write(data_bytes)

    def to_channel_ts(self, fn=None):
        """

        :param fn: DESCRIPTION, defaults to None
        :type fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if not self.has_metadata():
            msg = f"Could not find Metronix metadata file for {self.fn.name}."
            logger.warning(msg)
            ch_metadata = None
            return ChannelTS(data=self.read_atss())
        else:
            ch_metadata = MetronixChannelJSON(self.metadata_fn)
            return ChannelTS(
                channel_type=ch_metadata.type, data=self.read_atss()
            )


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
