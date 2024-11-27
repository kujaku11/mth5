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

from mth5.io.metronix import MetronixFileNameMetadata, MetronixChannelJSON

# =============================================================================


class ATSS:
    def __init__(self, **kwargs):
        pass


# 099_ADU-07e_C004_THz_32Hz .atss or .json
# tags are separated by "_"
# a tag can NOT contain a "_" because that is the separator!
# DO NOT USE UNDERSCORE IN TAGS!!!
# a system like ADU_02 is FORBIDDEN; use ADU-02 instead
def file_tags():
    # file name is part of the data - will NOT be repeated in header
    # order of the dictionary is file name!
    #
    file = {
        "serial": 0,  # such as 1234 (no negative numbers please) for the system
        "system": "",  # such as ADU-08e, XYZ (a manufacturer is not needed because the system indicates it)
        "channel_no": 0,  # channel number - you can integrate EAMP stations as channels if the have the SAME!! timings
        "channel_type": "",  # type such as Ex, Ey, Hx, Hy, Hz or currents like Jx, Jy, Jz or Pitch, Roll, Yaw or x, y, z or T for temperature
        "sample_rate": 0.0,  # contains sample_rate. Unit: Hz (samples per second) - "Hz" or "s" will be appended while writing in real time
        # if file name contains 4s it is 0.25Hz; if it contains 256Hz it is 256Hz
        # the FILENAME contains a UNIT for better readability; you MUST have 256Hz (sample rate 256) OR 4s (sample rate 0.25);
        # a "." in the FILENAME is possible on modern systems, 16.6666Hz is possible
        # the suffix .json or .atss is NOT part of the filename, append it while writing
    }
    return file


# ##################################################################################################################
def header():
    # contains items needed for a complete channel description together with calibration data
    # does NOT contain values from file name
    header = {
        # the Z suffix is mostly not supported by C/C++/Python/PHP and others
        "datetime": "1970-01-01T00:00:00.0",  # ISO 8601 datetime in UTC
        "latitude": 0.0,  # decimal degree such as 52.2443
        "longitude": 0.0,  # decimal degree such as 10.5594
        "elevation": 0.0,  # elevation in meter
        "azimuth": 0.0,  # orientation from North to East (90 = East, -90 or 270 = West, 180 South, 0 North)
        "tilt": 0.0,  # azimuth positive down - in case it had been measured
        "resistance": 0.0,  # resistance of the sensor in Ohm or contact resistance of electrode in Ohm
        "units": "mV",  # for ADUs it will be mV H or other -  or scaled E mV/km (the logger will do this while recording)
        "filter": "",  # comma separated list of filters such as "ADB-LF,LF-RF-4" or "ADB-HF,HF-RF-1"
        "source": "",  # empty or indicate as, ns, ca, cp, tx or what ever; some users need this
    }
    return header


# ##################################################################################################################

# finally a channel combines all the above


def channel():
    f = file_tags()  # file tags are part of the file name, NOT in json header
    h = header()
    channel = f | h  # merge dictionaries
    channel["sensor_calibration"] = (
        calibration()
    )  # needed at least for the sensor name and serial
    return channel


# ##################################################################################################################
# helper function to convert the sample rate to a string ending with Hz or s, you can give the precision
# avoid 265.0001Hz, 256.98 -> 257Hz or 4.0001 -> 4s rounding errors
# precision is the number of digits after the comma, default is 0, you get only full Hertz or seconds
# ATTENTION: the railway frequency is 16.6666Hz, you get 17 Hz then; take precision = 2 to get 16.67Hz
def sample_rate_to_string(sample_rate, precision=0):
    if sample_rate == 0.0:
        return "failed__zero_sample_rate"

    if precision == 0:
        if sample_rate > 0.5:  # assumed to be an rounding error
            fd = sample_rate - int(sample_rate)  # get the decimal part
            fi = int(sample_rate)  # get the integer part
            if fd > 0.9:
                fi = fi + 1
            return str(fi) + "Hz"
        else:
            fd = 1.0 / sample_rate - int(
                1.0 / sample_rate
            )  # get the decimal part
            fi = int(1.0 / sample_rate)  # get the integer part
            if fd > 0.5:
                fi = fi + 1
            return str(fi) + "s"
    else:
        if (
            sample_rate > 0.999
        ):  # you may want to correct the rounding error manually, I set to Hz
            return "{:.{prec}f}Hz".format(sample_rate, prec=precision)
        else:
            return "{:.{prec}f}s".format(1.0 / sample_rate, prec=precision)


# ##################################################################################################################
# returns a filename WITHOUT extension
# prepend_dir is the directory where the file is stored - a convenience function
# precision is the number of digits after the comma, default is 0, you get only full Hertz or seconds
def base_name(channel, precision=0, prepend_dir=""):
    array_tags = list(file_tags())  # these are the file keys
    filename = ""
    count = -1
    fill = ""
    for tag in array_tags:
        count = count + 1
        if count > 0:
            fill = "_"
        for key, value in channel.items():
            if tag == key:
                if tag == "sample_rate":
                    filename = (
                        filename
                        + fill
                        + sample_rate_to_string(
                            channel["sample_rate"], precision
                        )
                    )
                elif tag == "channel_no":
                    filename = (
                        filename + fill + "C" + f"{channel['channel_no']:03}"
                    )
                elif tag == "channel_type":
                    filename = (
                        filename + fill + "T" + f"{channel['channel_type']}"
                    )
                elif tag == "serial":
                    filename = filename + fill + f"{channel['serial']:03}"
                else:
                    filename = filename + fill + channel[tag]
    if prepend_dir != "":
        # use os.path.join to make it platform independent
        filename = os.path.join(prepend_dir, filename)
    return filename


# ##################################################################################################################
# write the json header file
# the file tags are separated by "_" and will be removed from the dictionary
# they are part of the file name, not the json header
def write_header(channel, file_name):
    file_items = file_tags()
    # temporary channel without the file items
    temp_channel = copy.deepcopy(channel)
    for item in file_items:
        temp_channel.pop(item)  # remove the header items and write json
    # write the json header file
    with open(atss_basename(file_name) + ".json", "w") as f:
        f.write(
            json.dumps(
                temp_channel, indent=2, sort_keys=False, ensure_ascii=False
            )
        )
        f.close()


# ##################################################################################################################
# the data array is a numpy array of type np.float64 and written as binary file
def write_data(data_array, file_name):
    with open(atss_basename(file_name) + ".atss", "wb") as f:
        # Convert the data array to bytes
        data_bytes = data_array.tobytes()
        # Write the bytes to the binary file
        f.write(data_bytes)
        f.close()


# ##################################################################################################################
# read the json header file
# the tags are separated by "_" and will be mapped to the dictionary
def read_header(file_name):
    # create an empty channel
    chan = channel()
    tagname = atss_basename(file_name)
    # strip the leading path, us os.path.basename instead
    tagname = os.path.basename(tagname)

    tags = tagname.split("_")
    chan["serial"] = int(tags[0])  # no leading tag indicator
    chan["system"] = tags[1]  # no leading tag indicator
    tags.pop(0)
    tags.pop(0)
    for tag in tags:  # the rest of the tags
        if tag.startswith("C"):  # channel number
            tag = tag[1:]
            chan["channel_no"] = int(tag)
        if tag.startswith("T"):  # channel type
            tag = tag[1:]
            chan["channel_type"] = tag
        # sample rate
        if tag.endswith("s") and tag[0].isdigit():
            tag = tag[:-1]
            fl = float(tag)
            chan["sample_rate"] = 1.0 / fl

        if tag.endswith("Hz") and tag[0].isdigit():
            tag = tag[:-2]
            chan["sample_rate"] = float(tag)

    with open(atss_basename(file_name) + ".json", "r") as f:
        header = json.load(f)
        f.close()
    # merge the header with the file name
    chan.update(header)
    return chan


# ##################################################################################################################
# read the data file
# run exists_both before to check if the files exist first
def read_data(file_name, start=0, wl=0):
    samples = os.path.getsize(atss_basename(file_name) + ".atss") / 8
    if start + wl > samples:
        raise ValueError(f"start + wl > samples")
    with open(atss_basename(file_name) + ".atss", "rb") as f:
        # Read the binary data
        f.seek(start * 8)
        if wl == 0:
            data_bytes = f.read()  # complete file
        else:
            data_bytes = f.read((wl) * 8)
        # Convert the data to a numpy array
        data_array = np.frombuffer(data_bytes, dtype=np.float64)
        f.close()
    return data_array


# ##################################################################################################################


# remove .json or .atss from the file name
def atss_basename(file_name):
    if file_name.endswith(".atss"):
        return file_name[:-5]
    elif file_name.endswith(".json"):
        return file_name[:-5]
    else:
        return file_name


# ##################################################################################################################


def samples(file_name):
    samples = os.path.getsize(atss_basename(file_name) + ".atss") / 8
    return int(samples)


# ##################################################################################################################


def stop_date_time(file_name):
    nsamples = samples(file_name)
    # get the sample_rate from the file name
    channel = read_header(file_name)
    # get the sample rate, the read_header function returns ensures that the sample rate is in Hz
    sample_rate = channel["sample_rate"]
    # get the start date time ISO 8601 like "1970-01-01T00:00:00.0"
    start_date_time = channel["datetime"]
    # calculate the stop date time
    stop_date_time = np.datetime64(start_date_time) + np.timedelta64(
        int(nsamples / sample_rate), "s"
    )
    return stop_date_time


def duration(file_name):
    # get the start date time ISO 8601 like "1970-01-01T00:00:00.0"
    start_date_time = read_header(file_name)["datetime"]
    # get the stop date time ISO 8601 like "1970-01-01T00:00:00.0"
    stop_date_time = stop_date_time(file_name)
    # calculate the duration
    duration = stop_date_time - np.datetime64(start_date_time)
    # return the duration in HH:MM:SS
    return str(duration)


# check if atss and json exist, and return the samples
def exits_both(file_name):
    sfile_name = atss_basename(file_name) + ".json"
    # if not exist, terminate with FileNotFoundError
    if not os.path.exists(sfile_name):
        raise FileNotFoundError(f"File {sfile_name} not found")
    #
    sfile_name = atss_basename(file_name) + ".atss"
    # if not exist, terminate with FileNotFoundError
    if not os.path.exists(sfile_name):
        raise FileNotFoundError(f"File {sfile_name} not found")
    # if both exist, return the amount samples
    return samples(file_name)


def cal_mfs_06e(spc, file_name, wl):
    # the calibration data for the MFS-06e sensor
    # the spc is the complex spectrum, calculated by the fft "backward" function
    # file_name is the file name of the channel, we take the sample rate from the header
    #
    # get the channel from the file
    if not os.path.exists(file_name + ".json"):
        raise FileNotFoundError(f"File {file_name}.json not found")
    channel = read_header(file_name)
    fs = channel["sample_rate"]
    chopper = channel["sensor_calibration"]["chopper"]
    if chopper == 1:
        # calculate the frequency for each bin
        for i, x in enumerate(spc):
            if i == 0:
                continue
            f = i * fs / wl
            p1 = complex(0.0, (f / 4.0))
            p2 = complex(0.0, (f / 8192.0))
            p4 = complex(0.0, (f / 28300.0))
            trf = 800.0 * (
                (p1 / (1.0 + p1)) * (1.0 / (1.0 + p2)) * (1.0 / (1.0 + p4))
            )
            spc[i] = spc[i] / trf
    else:
        # calculate the frequency for each bin
        for i in enumerate(spc):
            if i == 0:
                continue
            f = i * fs / wl
            p1 = complex(0.0, (f / 4.0))
            p2 = complex(0.0, (f / 8192.0))
            p3 = complex(0.0, (f / 0.720))
            p4 = complex(0.0, (f / 28300.0))
            trf = 800.0 * (
                (p1 / (1.0 + p1))
                * (1.0 / (1.0 + p2))
                * (p3 / (1.0 + p3))
                * (1.0 / (1.0 + p4))
            )
            spc[i] = spc[i] / trf
