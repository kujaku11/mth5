# -*- coding: utf-8 -*-
"""
===============
NIMS
===============

    * deals with reading in NIMS DATA.BIN files
    
    This is a translation from Matlab codes written and edited by:
        * Anna Kelbert
        * Paul Bedrosian
        * Esteban Bowles-Martinez
        * Possibly others.
        
    I've tested it against a version, and it matches.  The data/GPS  gaps I
    still don't understand so for now the time series is just 
    made continuous and the number of missing seconds is clipped from the 
    end of the time series.
    
    .. note:: this only works for 8Hz data for now
    
    
:copyright:
    Jared Peacock (jpeacock@usgs.gov)
    
:license: 
    MIT
"""

# =============================================================================
# Imports
# =============================================================================
import struct
import datetime

import numpy as np
import pandas as pd

from mth5.io.nims.gps import GPS
from mth5.io.nims.header import NIMSHeader
from mth5.io.nims.response_filters import Response
from mth5 import timeseries

from mt_metadata.utils.mttime import MTime
from mt_metadata.timeseries import Station, Run, Electric, Magnetic, Auxiliary

# =============================================================================
# Exceptions
# =============================================================================


class NIMS(NIMSHeader):
    """
    NIMS Class will read in a NIMS DATA.BIN file.

    A fast way to read the binary files are to first read in the GPS strings,
    the third byte in each block as a character and parse that into valid
    GPS stamps.

    Then read in the entire data set as unsigned 8 bit integers and reshape
    the data to be n seconds x block size.  Then parse that array into the
    status information and data.

    I only have a limited amount of .BIN files to test so this will likely
    break if there are issues such as data gaps.  This has been tested against the
    matlab program loadNIMS by Anna Kelbert and the match for all the .bin files
    I have.  If something looks weird check it against that program.

    .. todo:: deal with timing issues, right now a warning is sent to the user
              need to figure out a way to find where the gap is and adjust
              accordingly.

    .. warning::
        Currently Only 8 Hz data is supported

    """

    def __init__(self, fn=None):
        super().__init__(fn)

        # change thes if the sample rate is different
        self.block_size = 131
        self.block_sequence = [1, self.block_size]
        self.sample_rate = 8  ### samples/second
        self.e_conversion_factor = 2.44141221047903e-06
        self.h_conversion_factor = 0.01
        self.t_conversion_factor = 70
        self.t_offset = 18048
        self._int_max = 8388608
        self._int_factor = 16777216
        self._block_dict = {
            "soh": 0,
            "block_len": 1,
            "status": 2,
            "gps": 3,
            "sequence": 4,
            "box_temp": (5, 6),
            "head_temp": (7, 8),
            "logic": 81,
            "end": 130,
        }
        self.info_array = None
        self.stamps = None
        self.ts_data = None
        self.gaps = None
        self.duplicate_list = None

        self._raw_string = None

        self.indices = self._make_index_values()

    def __str__(self):
        lines = [f"NIMS Station: {self.site_name}", "-" * 20]
        lines.append(f"NIMS ID:         {self.box_id}")
        lines.append(f"magnetometer ID: {self.mag_id}")
        lines.append(f"operator:        {self.operator}")
        lines.append(f"location:        {self.state_province}, {self.country}")
        lines.append(f"latitude:        {self.latitude} (degrees)")
        lines.append(f"longitude:       {self.longitude} (degrees)")
        lines.append(f"elevation:       {self.elevation} m")
        lines.append(f"gps stamp:       {self.header_gps_stamp}")
        lines.append(f"EX: length = {self.ex_length} m; azimuth = {self.ex_azimuth}")
        lines.append(f"EY: length = {self.ey_length} m; azimuth = {self.ey_azimuth}")
        lines.append(f"comments:        {self.comments}")

        if self.has_data():
            lines.append("")
            lines.append(f"Start:      {self.start_time.isoformat()}")
            lines.append(f"End:        {self.end_time.isoformat()}")
            lines.append(f"Data shape: {self.ts_data.shape}")
            lines.append(f"Found {len(self.stamps)} GPS stamps")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def has_data(self):
        if self.ts_data is not None:
            return True
        return False

    @property
    def n_samples(self):
        if self.has_data():
            return self.ts_data.shape[0]
        elif self.fn is not None:
            return int(self.file_size / 16.375)

    @property
    def latitude(self):
        """
        median latitude value from all the GPS stamps in decimal degrees
        WGS84

        Only get from the GPRMC stamp as they should be duplicates
        """
        if self.stamps is not None:
            latitude = np.zeros(len(self.stamps))
            for ii, stamp in enumerate(self.stamps):
                latitude[ii] = stamp[1][0].latitude
            return np.median(latitude[np.nonzero(latitude)])
        return self.header_gps_latitude

    @property
    def longitude(self):
        """
        median longitude value from all the GPS stamps in decimal degrees
        WGS84

        Only get from the first stamp within the sets
        """
        if self.stamps is not None:
            longitude = np.zeros(len(self.stamps))
            for ii, stamp in enumerate(self.stamps):
                longitude[ii] = stamp[1][0].longitude
            return np.median(longitude[np.nonzero(longitude)])
        return self.header_gps_longitude

    @property
    def elevation(self):
        """
        median elevation value from all the GPS stamps in decimal degrees
        WGS84

        Only get from the first stamp within the sets
        """
        if self.stamps is not None:
            elevation = np.zeros(len(self.stamps))
            for ii, stamp in enumerate(self.stamps):
                if len(stamp[1]) == 1:
                    elev = stamp[1][0].elevation
                if len(stamp[1]) == 2:
                    elev = stamp[1][1].elevation
                if elev is None:
                    continue
                elevation[ii] = elev
            return np.median(elevation[np.nonzero(elevation)])
        return self.header_gps_elevation

    @property
    def declination(self):
        """
        median elevation value from all the GPS stamps in decimal degrees
        WGS84

        Only get from the first stamp within the sets
        """
        if self.stamps is not None:
            declination = np.zeros(len(self.stamps))
            for ii, stamp in enumerate(self.stamps):
                if stamp[1][0].gps_type == "GPRMC":
                    dec = stamp[1][0].declination
                if dec is None:
                    continue
                declination[ii] = dec
            return np.median(declination[np.nonzero(declination)])
        return None

    @property
    def start_time(self):
        """
        start time is the first good GPS time stamp minus the seconds to the
        beginning of the time series.
        """
        if self.stamps is not None:
            return MTime(self.ts_data.index[0])
        return self.header_gps_stamp

    @property
    def end_time(self):
        """
        start time is the first good GPS time stamp minus the seconds to the
        beginning of the time series.
        """
        if self.stamps is not None:
            return MTime(self.ts_data.index[-1])
        self.logger.warning("Estimating end time from n_samples")
        return self.start_time + int(self.n_samples / self.sample_rate)

    @property
    def box_temperature(self):
        """data logger temperature, sampled at 1 second"""

        if self.ts_data is not None:
            auxiliary_metadata = Auxiliary()
            auxiliary_metadata.from_dict(
                {
                    "channel_number": 6,
                    "component": "temperature",
                    "measurement_azimuth": 0,
                    "measurement_tilt": 0,
                    "sample_rate": 1,
                    "time_period.start": self.start_time.isoformat(),
                    "time_period.end": self.end_time.isoformat(),
                    "type": "auxiliary",
                    "units": "celsius",
                }
            )

            temp = timeseries.ChannelTS(
                channel_type="auxiliary",
                data=self.info_array["box_temp"],
                channel_metadata=auxiliary_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
            # interpolate temperature onto the same sample rate as the channels.
            temp.data_array = temp.data_array.interp_like(self.hx.data_array)
            temp.channel_metadata.sample_rate = self.sample_rate
            temp.channel_metadata.time_period.end = self.end_time.isoformat()

            return temp
        return None

    def get_channel_response(self, channel, dipole_length=1):
        """
        Get the channel response for a given channel

        :param channel: DESCRIPTION
        :type channel: TYPE
        :param dipole_length: DESCRIPTION, defaults to 1
        :type dipole_length: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        nims_filters = Response(sample_rate=self.sample_rate)
        return nims_filters.get_channel_response(channel, dipole_length=dipole_length)

    @property
    def hx_metadata(self):
        if self.ts_data is not None:
            hx_metadata = Magnetic()
            hx_metadata.from_dict(
                {
                    "channel_number": 1,
                    "component": "hx",
                    "measurement_azimuth": 0,
                    "measurement_tilt": 0,
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start_time.isoformat(),
                    "time_period.end": self.end_time.isoformat(),
                    "type": "magnetic",
                    "units": "counts",
                    "sensor.id": self.mag_id,
                    "sensor.manufacturer": "Barry Narod",
                    "sensor.type": "fluxgate triaxial magnetometer",
                }
            )
            return hx_metadata

    @property
    def hx(self):
        """HX"""
        if self.ts_data is not None:

            return timeseries.ChannelTS(
                channel_type="magnetic",
                data=self.ts_data.hx.to_numpy(),
                channel_metadata=self.hx_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                channel_response=self.get_channel_response("hx"),
            )
        return None

    @property
    def hy_metadata(self):
        if self.ts_data is not None:
            hy_metadata = Magnetic()
            hy_metadata.from_dict(
                {
                    "channel_number": 2,
                    "component": "hy",
                    "measurement_azimuth": 90,
                    "measurement_tilt": 0,
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start_time.isoformat(),
                    "time_period.end": self.end_time.isoformat(),
                    "type": "magnetic",
                    "units": "counts",
                    "sensor.id": self.mag_id,
                    "sensor.manufacturer": "Barry Narod",
                    "sensor.type": "fluxgate triaxial magnetometer",
                }
            )
            return hy_metadata

    @property
    def hy(self):
        """HY"""
        if self.ts_data is not None:
            return timeseries.ChannelTS(
                channel_type="magnetic",
                data=self.ts_data.hy.to_numpy(),
                channel_metadata=self.hy_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                channel_response=self.get_channel_response("hy"),
            )
        return None

    @property
    def hz_metadata(self):
        if self.ts_data is not None:
            hz_metadata = Magnetic()
            hz_metadata.from_dict(
                {
                    "channel_number": 3,
                    "component": "hz",
                    "measurement_azimuth": 0,
                    "measurement_tilt": 0,
                    "sample_rate": self.sample_rate,
                    "time_period.start": self.start_time.isoformat(),
                    "time_period.end": self.end_time.isoformat(),
                    "type": "magnetic",
                    "units": "counts",
                    "sensor.id": self.mag_id,
                    "sensor.manufacturer": "Barry Narod",
                    "sensor.type": "fluxgate triaxial magnetometer",
                }
            )
            return hz_metadata

    @property
    def hz(self):
        """HZ"""
        if self.ts_data is not None:

            return timeseries.ChannelTS(
                channel_type="magnetic",
                data=self.ts_data.hz.to_numpy(),
                channel_metadata=self.hz_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                channel_response=self.get_channel_response("hz"),
            )
        return None

    @property
    def ex_metadata(self):
        if self.ts_data is not None:
            ex_metadata = Electric()
            ex_metadata.from_dict(
                {
                    "channel_number": 4,
                    "component": "ex",
                    "measurement_azimuth": self.ex_azimuth,
                    "measurement_tilt": 0,
                    "sample_rate": self.sample_rate,
                    "dipole_length": self.ex_length,
                    "time_period.start": self.start_time.isoformat(),
                    "time_period.end": self.end_time.isoformat(),
                    "type": "electric",
                    "units": "counts",
                    "negative.id": self.s_electrode_id,
                    "positive.id": self.n_electrode_id,
                }
            )

            return ex_metadata

    @property
    def ex(self):
        """EX"""
        if self.ts_data is not None:

            return timeseries.ChannelTS(
                channel_type="electric",
                data=self.ts_data.ex.to_numpy(),
                channel_metadata=self.ex_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                channel_response=self.get_channel_response("ex", self.ex_length),
            )
        return None

    @property
    def ey_metadata(self):
        if self.ts_data is not None:
            ey_metadata = Electric()
            ey_metadata.from_dict(
                {
                    "channel_number": 5,
                    "component": "ey",
                    "measurement_azimuth": self.ey_azimuth,
                    "measurement_tilt": 0,
                    "sample_rate": self.sample_rate,
                    "dipole_length": self.ey_length,
                    "time_period.start": self.start_time.isoformat(),
                    "time_period.end": self.end_time.isoformat(),
                    "type": "electric",
                    "units": "counts",
                    "negative.id": self.w_electrode_id,
                    "positive.id": self.e_electrode_id,
                }
            )

            return ey_metadata

    @property
    def ey(self):
        """EY"""
        if self.ts_data is not None:

            return timeseries.ChannelTS(
                channel_type="electric",
                data=self.ts_data.ey.to_numpy(),
                channel_metadata=self.ey_metadata,
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
                channel_response=self.get_channel_response("ey", self.ey_length),
            )
        return None

    @property
    def run_metadata(self):
        """Run metadata"""

        if self.ts_data is not None:
            run_metadata = Run()
            run_metadata.from_dict(
                {
                    "Run": {
                        "comments": self.comments,
                        "data_logger.firmware.author": "B. Narod",
                        "data_logger.firmware.name": "nims",
                        "data_logger.firmware.version": "1.0",
                        "data_logger.manufacturer": "Narod",
                        "data_logger.model": self.box_id,
                        "data_logger.id": self.box_id,
                        "data_logger.type": "long period",
                        "id": self.run_id,
                        "data_type": "MTLP",
                        "sample_rate": self.sample_rate,
                        "time_period.end": self.end_time.isoformat(),
                        "time_period.start": self.start_time.isoformat(),
                    }
                }
            )
            for comp in ["hx", "hy", "hz", "ex", "ey"]:
                run_metadata.channels.append(getattr(self, f"{comp}_metadata"))
            return run_metadata
        return None

    @property
    def station_metadata(self):
        """Station metadata from nims file"""
        if self.ts_data is not None:
            station_metadata = Station()
            station_metadata.from_dict(
                {
                    "Station": {
                        "geographic_name": f"{self.site_name}, {self.state_province}, {self.country}",
                        "location.declination.value": self.declination,
                        "location.elevation": self.elevation,
                        "location.latitude": self.latitude,
                        "location.longitude": self.longitude,
                        "id": self.run_id[0:-1],
                        "orientation.reference_frame": "geomagnetic",
                    }
                }
            )
            station_metadata.runs.append(self.run_metadata)
            return station_metadata
        return None

    def to_runts(self, calibrate=False):
        """Get xarray for run"""

        if self.ts_data is not None:
            run = timeseries.RunTS(
                array_list=[
                    self.hx,
                    self.hy,
                    self.hz,
                    self.ex,
                    self.ey,
                    self.box_temperature,
                ],
                run_metadata=self.run_metadata,
                station_metadata=self.station_metadata,
            )
            if calibrate:
                return run.calibrate()
            else:
                return run
        return None

    def _make_index_values(self):
        """
        Index values for the channels recorded
        """
        ### make an array of index values for magnetics and electrics
        indices = np.zeros((8, 5), dtype=int)
        for kk in range(8):
            ### magnetic blocks
            for ii in range(3):
                indices[kk, ii] = 9 + (kk) * 9 + (ii) * 3
            ### electric blocks
            for ii in range(2):
                indices[kk, 3 + ii] = 82 + (kk) * 6 + (ii) * 3
        return indices

    def _get_gps_string_list(self, nims_string):
        """
        get the gps strings from the raw string output by the NIMS.  This will
        take the 3rd value in each block, concatenate into a long string and
        then make a list by splitting by '$'.  The index values of where the
        '$' are found are also calculated.

        :param str nims_string: raw binary string output by NIMS

        :returns: list of index values associated with the location of the '$'

        :returns: list of possible raw GPS strings

        .. note:: This assumes that there are an even amount of data blocks.
                  Might be a bad assumption
        """
        ### get index values of $ and gps_strings
        index_values = []
        gps_str_list = []
        for ii in range(int(len(nims_string) / self.block_size)):
            index = ii * self.block_size + 3
            g_char = struct.unpack("c", nims_string[index : index + 1])[0]
            if g_char == b"$":
                index_values.append((index - 3) / self.block_size)
            gps_str_list.append(g_char)
        gps_raw_stamp_list = b"".join(gps_str_list).split(b"$")
        return index_values, gps_raw_stamp_list

    def get_stamps(self, nims_string):
        """
        get a list of valid GPS strings and match synchronous GPRMC with GPGGA
        stamps if possible.

        :param str nims_string: raw GPS string output by NIMS
        """
        ### read in GPS strings into a list to be parsed later
        index_list, gps_raw_stamp_list = self._get_gps_string_list(nims_string)

        gprmc_list = []
        gpgga_list = []
        ### note we are skipping the first entry, it tends to be not
        ### complete anyway
        for ii, index, raw_stamp in zip(
            range(len(index_list)), index_list, gps_raw_stamp_list[1:]
        ):
            gps_obj = GPS(raw_stamp, index)
            if gps_obj.valid:
                if gps_obj.gps_type == "GPRMC":
                    gprmc_list.append(gps_obj)
                elif gps_obj.gps_type == "GPGGA":
                    gpgga_list.append(gps_obj)
            else:
                self.logger.debug(f"GPS Error: file index {index}, stamp number {ii}")
                max_len = min([len(raw_stamp), 15])
                self.logger.debug(f"GPS Raw Stamp: {raw_stamp[0:max_len]}")
        return self._gps_match_gprmc_gpgga_strings(gprmc_list, gpgga_list)

    def _gps_match_gprmc_gpgga_strings(self, gprmc_list, gpgga_list):
        """
        match GPRMC and GPGGA strings together into a list

        [[GPRMC, GPGGA], ...]

        :param list gprmc_list: list of GPS objects for the GPRMC stamps
        :param list gpgga_list: list of GPS objects for the GPGGA stamps

        :returns: list of matched GPRMC and GPGGA stamps

        """
        ### match up the GPRMC and GPGGA together
        gps_match_list = []
        for gprmc in gprmc_list:
            find = False
            for ii, gpgga in enumerate(gpgga_list):
                if gprmc.time_stamp.time() == gpgga.time_stamp.time():
                    gps_match_list.append([gprmc, gpgga])
                    find = True
                    del gpgga_list[ii]
                    break
            if not find:
                gps_match_list.append([gprmc])
        return gps_match_list

    def _get_gps_stamp_indices_from_status(self, status_array):
        """
        get the index location of the stamps from the status array assuming
        that 0 indicates GPS lock.

        :param :class:`np.ndarray` status_array: an array of status values from data blocks

        :returns: array of index values where GPS lock was acquired ignoring
                  sequential locks.
        """

        index_values = np.where(status_array == 0)[0]
        status_index = np.zeros_like(index_values)
        for ii in range(index_values.size):
            if index_values[ii] - index_values[ii - 1] == 1:
                continue
            else:
                status_index[ii] = index_values[ii]
        status_index = status_index[np.nonzero(status_index)]

        return status_index

    def match_status_with_gps_stamps(self, status_array, gps_list):
        """
        Match the index values from the status array with the index values of
        the GPS stamps.  There appears to be a bit of wiggle room between when the
        lock is recorded and the stamp was actually recorded.  This is typically 1
        second and sometimes 2.

        :param array status_array: array of status values from each data block
        :param list gps_list: list of valid GPS stamps [[GPRMC, GPGGA], ...]

        .. note:: I think there is a 2 second gap between the lock and the
                  first stamp character.
        """

        stamp_indices = self._get_gps_stamp_indices_from_status(status_array)
        gps_stamps = []
        for index in stamp_indices:
            stamp_find = False
            for ii, stamps in enumerate(gps_list):
                index_diff = stamps[0].index - index
                ### check the index value, should be 2 or 74, if it is off by
                ### a value left or right apply a correction.
                if index_diff == 1 or index_diff == 73:
                    index += 1
                    stamps[0].index += 1
                elif index_diff == 2 or index_diff == 74:
                    index = index
                elif index_diff == 3 or index_diff == 75:
                    index -= 1
                    stamps[0].index -= 1
                elif index_diff == 4 or index_diff == 76:
                    index -= 2
                    stamps[0].index -= 2
                if stamps[0].gps_type in ["GPRMC", "gprmc"]:
                    if index_diff in [1, 2, 3, 4]:
                        gps_stamps.append((index, stamps))
                        stamp_find = True
                        del gps_list[ii]
                        break
                elif stamps[0].gps_type in ["GPGGA", "gpgga"]:
                    if index_diff in [73, 74, 75, 76]:
                        gps_stamps.append((index, stamps))
                        stamp_find = True
                        del gps_list[ii]
                        break
            if not stamp_find:
                self.logger.debug(f"GPS Error: No good GPS stamp at {index} seconds")
        return gps_stamps

    def find_sequence(self, data_array, block_sequence=None):
        """
        find a sequence in a given array

        :param array data_array: array of the data with shape [n, m]
                                 where n is the number of seconds recorded
                                 m is the block length for a given sampling
                                 rate.
        :param list block_sequence: sequence pattern to locate
                                    *default* is [1, 131] the start of a
                                    data block.

        :returns: array of index locations where the sequence is found.
        """
        if block_sequence is not None:
            self.block_sequence = block_sequence
        # want to find the index there the test data is equal to the test sequence
        t = np.vstack(
            [
                np.roll(data_array, shift)
                for shift in -np.arange(len(self.block_sequence))
            ]
        ).T
        return np.where(np.all(t == self.block_sequence, axis=1))[0]

    def unwrap_sequence(self, sequence):
        """
        unwrap the sequence to be sequential numbers instead of modulated by
        256.  sets the first number to 0

        :param list sequence: sequence of bytes numbers
        :return: unwrapped number of counts

        """
        count = 0
        unwrapped = np.zeros_like(sequence)
        for ii, seq in enumerate(sequence):
            unwrapped[ii] = seq + count * 256
            if seq == 255:
                count += 1
        unwrapped -= unwrapped[0]

        return unwrapped

    def _locate_duplicate_blocks(self, sequence):
        """
        locate the sequence number where the duplicates exist

        :param list sequence: sequence to match duplicate numbers.
        :returns: list of duplicate index values.
        """

        duplicates = np.where(np.abs(np.diff(sequence)) == 0)[0]
        if len(duplicates) == 0:
            return None
        duplicate_list = []
        for dup in duplicates:
            dup_dict = {}
            dup_dict["sequence_index"] = dup
            dup_dict["ts_index_0"] = dup * self.sample_rate
            dup_dict["ts_index_1"] = dup * self.sample_rate + self.sample_rate
            dup_dict["ts_index_2"] = (dup + 1) * self.sample_rate
            dup_dict["ts_index_3"] = (dup + 1) * self.sample_rate + self.sample_rate
            duplicate_list.append(dup_dict)
        return duplicate_list

    def _check_duplicate_blocks(self, block_01, block_02, info_01, info_02):
        """
        make sure the blocks are truly duplicates

        :param np.array block_01: block of data to compare
        :param np.array block_02: block of data to compare
        :param np.array info_01: information array from info_array[sequence_index]
        :param np.array info_02: information array from info_array[sequence_index]

        :returns: boolean if the blocks and information match

        """
        if np.array_equal(block_01, block_02):
            if np.array_equal(info_01, info_02):
                return True
            else:
                return False
        else:
            return False

    def remove_duplicates(self, info_array, data_array):
        """
        remove duplicate blocks, removing the first duplicate as suggested by
        Paul and Anna. Checks to make sure that the mag data are identical for
        the duplicate blocks.  Removes the blocks from the information and
        data arrays and returns the reduced arrays.  This should sync up the
        timing of GPS stamps and index values.

        :param np.array info_array: structured array of block information
        :param np.array data_array: structured array of the data

        :returns: reduced information array
        :returns: reduced data array
        :returns: index of duplicates in raw data

        """
        ### locate
        duplicate_test_list = self._locate_duplicate_blocks(self.info_array["sequence"])
        if duplicate_test_list is None:
            return info_array, data_array, None
        duplicate_list = []
        for d in duplicate_test_list:
            if self._check_duplicate_blocks(
                data_array[d["ts_index_0"] : d["ts_index_1"]],
                data_array[d["ts_index_2"] : d["ts_index_3"]],
                info_array[d["sequence_index"]],
                info_array[d["sequence_index"] + 1],
            ):
                duplicate_list.append(d)
        self.logger.debug(f"Deleting {len(duplicate_list)} duplicate blocks")
        ### get the index of the blocks to be removed, namely the 1st duplicate
        ### block
        remove_sequence_index = [d["sequence_index"] for d in duplicate_list]
        remove_data_index = np.array(
            [np.arange(d["ts_index_0"], d["ts_index_1"], 1) for d in duplicate_list]
        ).flatten()
        ### remove the data
        return_info_array = np.delete(info_array, remove_sequence_index)
        return_data_array = np.delete(data_array, remove_data_index)

        ### set sequence to be monotonic
        return_info_array["sequence"][:] = np.arange(return_info_array.shape[0])

        return return_info_array, return_data_array, duplicate_list

    def read_nims(self, fn=None):
        """
        Read NIMS DATA.BIN file.

        1. Read in the header information and stores those as attributes
           with the same names as in the header file.

        2. Locate the beginning of the data blocks by looking for the
           first [1, 131, ...] combo.  Anything before that is cut out.

        3. Make sure the data is a multiple of the block length, if the
           data is longer the extra bits are cut off.

        4. Read in the GPS data (3rd byte of each block) as characters.
           Parses those into valid GPS stamps with appropriate index locations
           of where the '$' was found.

        5. Read in the data as unsigned 8-bit integers and reshape the array
           into [N, data_block_length].  Parse this array into the status
           information and the data.

        6. Remove duplicate blocks, by removing the first of the duplicates
           as suggested by Anna and Paul.

        7. Match the GPS locks from the status with valid GPS stamps.

        8. Check to make sure that there is the correct number of seconds
           between the first and last GPS stamp.  The extra seconds are cut
           off from the end of the time series.  Not sure if this is the
           best way to accommodate gaps in the data.

        .. note:: The data and information array returned have the duplicates
                  removed and the sequence reset to be monotonic.

        :param str fn: full path to DATA.BIN file

        :Example:

        >>> from mth5.io import nims
        >>> n = nims.NIMS(r"/home/mt_data/nims/mt001.bin")


        """

        if fn is not None:
            self.fn = fn
        st = datetime.datetime.now()
        ### read in header information and get the location of end of header
        self.read_header(self.fn)

        ### load in the entire file, its not too big, start from the
        ### end of the header information.
        with open(self.fn, "rb") as fid:
            fid.seek(self.data_start_seek)
            self._raw_string = fid.read()
        ### read in full string as unsigned integers
        data = np.frombuffer(self._raw_string, dtype=np.uint8)

        ### need to make sure that the data starts with a full block
        find_first = self.find_sequence(data[0 : self.block_size * 5])[0]
        data = data[find_first:]

        ### get GPS stamps from the binary string first
        self.gps_list = self.get_stamps(self._raw_string[find_first:])

        ### check the size of the data, should have an equal amount of blocks
        if (data.size % self.block_size) != 0:
            self.logger.warning(
                f"odd number of bytes {data.size}, not even blocks "
                f"cutting down the data by {data.size % self.block_size} bits"
            )
            end_data = data.size - (data.size % self.block_size)
            data = data[0:end_data]
        # resized the data into an even amount of blocks
        data = data.reshape((int(data.size / self.block_size), self.block_size))

        ### need to parse the data
        ### first get the status information
        self.info_array = np.zeros(
            data.shape[0],
            dtype=[
                ("soh", int),
                ("block_len", int),
                ("status", int),
                ("gps", int),
                ("sequence", int),
                ("box_temp", float),
                ("head_temp", float),
                ("logic", int),
                ("end", int),
            ],
        )

        for key, index in self._block_dict.items():
            if "temp" in key:
                # compute temperature
                t_value = data[:, index[0]] * 256 + data[:, index[1]]

                # something to do with the bits where you have to subtract
                t_value[np.where(t_value > 32768)] -= 65536
                value = (t_value - self.t_offset) / self.t_conversion_factor
            else:
                value = data[:, index]
            self.info_array[key][:] = value
        ### unwrap sequence
        self.info_array["sequence"] = self.unwrap_sequence(self.info_array["sequence"])

        ### get data
        data_array = np.zeros(
            data.shape[0] * self.sample_rate,
            dtype=[
                ("hx", float),
                ("hy", float),
                ("hz", float),
                ("ex", float),
                ("ey", float),
            ],
        )

        ### fill the data
        for cc, comp in enumerate(["hx", "hy", "hz", "ex", "ey"]):
            channel_arr = np.zeros((data.shape[0], 8), dtype=float)
            for kk in range(self.sample_rate):
                index = self.indices[kk, cc]
                value = (data[:, index] * 256 + data[:, index + 1]) * np.array(
                    [256]
                ) + data[:, index + 2]
                value[np.where(value > self._int_max)] -= self._int_factor
                channel_arr[:, kk] = value
            data_array[comp][:] = channel_arr.flatten()
        ### clean things up
        ### I guess that the E channels are opposite phase?
        for comp in ["ex", "ey"]:
            data_array[comp] *= -1
        ### remove duplicates
        (
            self.info_array,
            data_array,
            self.duplicate_list,
        ) = self.remove_duplicates(self.info_array, data_array)
        ### get GPS stamps with index values
        self.stamps = self.match_status_with_gps_stamps(
            self.info_array["status"], self.gps_list
        )
        ### align data checking for timing gaps
        self.ts_data = self.align_data(data_array, self.stamps)

        et = datetime.datetime.now()
        read_time = (et - st).total_seconds()
        self.logger.debug(f"Reading took {read_time:.2f} seconds")

    def _get_first_gps_stamp(self, stamps):
        """
        get the first GPRMC stamp
        """
        for stamp in stamps:
            if stamp[1][0].gps_type in ["gprmc", "GPRMC"]:
                return stamp
        return None

    def _get_last_gps_stamp(self, stamps):
        """
        get the last gprmc stamp
        """
        for stamp in stamps[::-1]:
            if stamp[1][0].gps_type in ["gprmc", "GPRMC"]:
                return stamp
        return None

    def _locate_timing_gaps(self, stamps):
        """
        locate timing gaps in the data by comparing the stamp index with the
        GPS time stamp.  The number of points and seconds should be the same

        :param list stamps: list of GPS stamps [[status_index, [GPRMC, GPGGA]]]

        :returns: list of gap index values
        """
        stamp_01 = self._get_first_gps_stamp(stamps)[1][0]
        # current_gap = 0
        current_stamp = stamp_01
        gap_beginning = []
        total_gap = 0
        for ii, stamp in enumerate(stamps[1:], 1):
            stamp = stamp[1][0]
            # can only compare those with a date and time.
            if stamp.gps_type == "GPGGA":
                continue
            time_diff = (stamp.time_stamp - current_stamp.time_stamp).total_seconds()
            index_diff = stamp.index - current_stamp.index

            time_gap = index_diff - time_diff
            if time_gap == 0:
                continue
            elif time_gap > 0:
                total_gap += time_gap
                current_stamp = stamp
                gap_beginning.append(stamp.index)
                self.logger.debug(
                    f"GPS tamp at {stamp.time_stamp.isoformat()} is off "
                    f"from previous time by { time_gap} seconds"
                )
        self.logger.warning(f"Timing is off by {total_gap} seconds")
        return gap_beginning

    def check_timing(self, stamps):
        """
        make sure that there are the correct number of seconds in between
        the first and last GPS GPRMC stamps

        :param list stamps: list of GPS stamps [[status_index, [GPRMC, GPGGA]]]

        :returns: [ True | False ] if data is valid or not.
        :returns: gap index locations

        .. note:: currently it is assumed that if a data gap occurs the data can be
                  squeezed to remove them.  Probably a more elegant way of doing it.
        """
        gaps = None
        first_stamp = self._get_first_gps_stamp(stamps)[1][0]
        last_stamp = self._get_last_gps_stamp(stamps)[1][0]

        time_diff = last_stamp.time_stamp - first_stamp.time_stamp
        index_diff = last_stamp.index - first_stamp.index

        difference = index_diff - time_diff.total_seconds()
        if difference != 0:
            gaps = self._locate_timing_gaps(stamps)
            return False, gaps, difference
        return True, gaps, difference

    def align_data(self, data_array, stamps):
        """
        Need to match up the first good GPS stamp with the data

        Do this by using the first GPS stamp and assuming that the time from
        the first time stamp to the start is the index value.

        put the data into a pandas data frame that is indexed by time

        :param array data_array: structure array with columns for each
                                 component [hx, hy, hz, ex, ey]
        :param list stamps: list of GPS stamps [[status_index, [GPRMC, GPGGA]]]

        :returns: pandas DataFrame with colums of components and indexed by
                  time initialized by the start time.

        .. note:: Data gaps are squeezed cause not sure what a gap actually means.
        """
        ### check timing first to make sure there is no drift
        timing_valid, self.gaps, time_difference = self.check_timing(stamps)

        ### need to trim off the excess number of points that are present because of
        ### data gaps.  This will be the time difference times the sample rate
        if time_difference > 0:
            remove_points = int(time_difference * self.sample_rate)
            data_array = data_array[0:-remove_points]
            self.logger.info(
                f"Trimmed {remove_points} points off the end of the time "
                "series because of timing gaps"
            )
        ### first GPS stamp within the data is at a given index that is
        ### assumed to be the number of seconds from the start of the run.
        ### therefore make the start time the first GPS stamp time minus
        ### the index value for that stamp.
        ### need to be sure that the first GPS stamp has a date, need GPRMC
        first_stamp = self._get_first_gps_stamp(stamps)
        first_index = first_stamp[0]
        start_time = first_stamp[1][0].time_stamp - datetime.timedelta(
            seconds=int(first_index)
        )

        dt_index = self.make_dt_index(
            start_time.isoformat(),
            self.sample_rate,
            n_samples=data_array.shape[0],
        )

        return pd.DataFrame(data_array, index=dt_index)

    def make_dt_index(self, start_time, sample_rate, stop_time=None, n_samples=None):
        """
        make time index array

        .. note:: date-time format should be YYYY-M-DDThh:mm:ss.ms UTC

        :param start_time: start time
        :type start_time: string

        :param end_time: end time
        :type end_time: string

        :param sample_rate: sample_rate in samples/second
        :type sample_rate: float
        """

        # set the index to be UTC time
        dt_freq = "{0:.0f}N".format(1.0 / (sample_rate) * 1e9)
        if stop_time is not None:
            dt_index = pd.date_range(
                start=start_time,
                end=stop_time,
                freq=dt_freq,
                closed="left",
                tz="UTC",
            )
        elif n_samples is not None:
            dt_index = pd.date_range(
                start=start_time, periods=n_samples, freq=dt_freq, tz="UTC"
            )
        else:
            raise ValueError("Need to input either stop_time or n_samples")
        return dt_index


# =============================================================================
# convenience read
# =============================================================================
def read_nims(fn):
    """

    :param fn: DESCRIPTION
    :type fn: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    nims_obj = NIMS(fn)
    nims_obj.read_nims()

    return nims_obj.to_runts()
