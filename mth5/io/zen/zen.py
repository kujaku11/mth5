# -*- coding: utf-8 -*-
"""
====================
Zen
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

"""

# ==============================================================================

import datetime
import struct
from pathlib import Path

import numpy as np
from loguru import logger
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries import Electric, Magnetic, Run, Station
from mt_metadata.timeseries.filters import (
    ChannelResponse,
    CoefficientFilter,
    FrequencyResponseTableFilter,
)

from mth5.io.zen import Z3DHeader, Z3DMetadata, Z3DSchedule
from mth5.io.zen.coil_response import CoilResponse
from mth5.timeseries import ChannelTS


# ==============================================================================
#
# ==============================================================================
class Z3D:
    """
    Deals with the raw Z3D files output by zen.
    Arguments
    -----------
        **fn** : string
                 full path to .Z3D file to be read in
    ======================== ================================ =================
    Attributes               Description                      Default Value
    ======================== ================================ =================
    _block_len               length of data block to read in  65536
                             as chunks faster reading
    _counts_to_mv_conversion conversion factor to convert     9.53674316406e-10
                             counts to mv
    _gps_bytes               number of bytes for a gps stamp  16
    _gps_dtype               data type for a gps stamp        see below
    _gps_epoch               starting date of GPS time
                             format is a tuple                (1980, 1, 6, 0,
                                                               0, 0, -1, -1, 0)
    _gps_f0                  first gps flag in raw binary
    _gps_f1                  second gps flag in raw binary
    _gps_flag_0              first gps flag as an int32       2147483647
    _gps_flag_1              second gps flag as an int32      -2147483648
    _gps_stamp_length        bit length of gps stamp          64
    _leap_seconds            leap seconds, difference         16
                             between UTC time and GPS
                             time.  GPS time is ahead
                             by this much
    _week_len                week length in seconds           604800
    df                       sampling rate of the data        256
    fn                       Z3D file name                    None
    gps_flag                 full gps flag                    _gps_f0+_gps_f1
    gps_stamps               np.ndarray of gps stamps         None
    header                   Z3DHeader object                 Z3DHeader
    metadata                 Z3DMetadata                      Z3DMetadata
    schedule                 Z3DSchedule                      Z3DSchedule
    time_series              np.ndarra(len_data)              None
    units                    units in which the data is in    counts
    zen_schedule             time when zen was set to         None
                             run
    ======================== ================================ =================

    * gps_dtype is formated as np.dtype([('flag0', np.int32),
                                        ('flag1', np.int32),
                                        ('time', np.int32),
                                        ('lat', np.float64),
                                        ('lon', np.float64),
                                        ('num_sat', np.int32),
                                        ('gps_sens', np.int32),
                                        ('temperature', np.float32),
                                        ('voltage', np.float32),
                                        ('num_fpga', np.int32),
                                        ('num_adc', np.int32),
                                        ('pps_count', np.int32),
                                        ('dac_tune', np.int32),
                                        ('block_len', np.int32)])


    :Example:

        >>> import mtpy.usgs.zen as zen
        >>> zt = zen.Zen3D(r"/home/mt/mt00/mt00_20150522_080000_256_EX.Z3D")
        >>> zt.read_z3d()
        >>> ------- Reading /home/mt/mt00/mt00_20150522_080000_256_EX.Z3D -----
            --> Reading data took: 0.322 seconds
            Scheduled time was 2015-05-22,08:00:16 (GPS time)
            1st good stamp was 2015-05-22,08:00:18 (GPS time)
            difference of 2.00 seconds
            found 6418 GPS time stamps
            found 1642752 data points
        >>> zt.plot_time_series()
    """

    def __init__(self, fn=None, **kwargs):
        self.logger = logger
        self.fn = fn
        self.calibration_fn = None

        self.header = Z3DHeader(fn)
        self.schedule = Z3DSchedule(fn)
        self.metadata = Z3DMetadata(fn)

        self._gps_stamp_length = kwargs.pop("stamp_len", 64)
        self._gps_bytes = self._gps_stamp_length / 4

        self.gps_stamps = None

        self._gps_flag_0 = np.int32(2147483647)
        self._gps_flag_1 = np.int32(-2147483648)
        self._gps_f0 = self._gps_flag_0.tobytes()
        self._gps_f1 = self._gps_flag_1.tobytes()
        self.gps_flag = self._gps_f0 + self._gps_f1

        self._gps_dtype = np.dtype(
            [
                ("flag0", np.int32),
                ("flag1", np.int32),
                ("time", np.int32),
                ("lat", np.float64),
                ("lon", np.float64),
                ("gps_sens", np.int32),
                ("num_sat", np.int32),
                ("temperature", np.float32),
                ("voltage", np.float32),
                ("num_fpga", np.int32),
                ("num_adc", np.int32),
                ("pps_count", np.int32),
                ("dac_tune", np.int32),
                ("block_len", np.int32),
            ]
        )

        self._week_len = 604800
        # '1980, 1, 6, 0, 0, 0, -1, -1, 0
        self._gps_epoch = MTime(time_stamp="1980-01-06T00:00:00")
        self._leap_seconds = 18
        self._block_len = 2**16
        # the number in the cac files is for volts, we want mV
        self._counts_to_mv_conversion = 9.5367431640625e-10 * 1e3
        self.num_sec_to_skip = 2

        self.units = "counts"
        self.sample_rate = None
        self.time_series = None
        self._max_time_diff = 20

        self.ch_dict = {"hx": 1, "hy": 2, "hz": 3, "ex": 4, "ey": 5}

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, fn):
        if fn is not None:
            self._fn = Path(fn)
        else:
            self._fn = None

    @property
    def file_size(self):
        if self.fn is not None:
            return self.fn.stat().st_size
        return 0

    @property
    def n_samples(self):
        if self.time_series is None:
            if self.sample_rate:
                return int(
                    (self.file_size - 512 * (1 + self.metadata.count)) / 4
                    + 8 * self.sample_rate
                )
            else:
                # assume just the 3 general metadata blocks
                return int((self.file_size - 512 * 3) / 4)
        else:
            return self.time_series.size

    @property
    def station(self):
        """
        station name
        """
        return self.metadata.station

    @station.setter
    def station(self, station):
        """
        station name
        """
        self.metadata.station = station

    @property
    def dipole_length(self):
        """
        dipole length
        """
        length = 0
        if self.metadata.ch_length is not None:
            length = float(self.metadata.ch_length)
        elif hasattr(self.metadata, "ch_offset_xyz1"):
            # only ex and ey have xyz2
            if hasattr(self.metadata, "ch_offset_xyz2"):
                x1, y1, z1 = [
                    float(offset) for offset in self.metadata.ch_offset_xyz1.split(":")
                ]
                x2, y2, z2 = [
                    float(offset) for offset in self.metadata.ch_offset_xyz2.split(":")
                ]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                length = np.round(length, 2)
            else:
                length = 0
        elif self.metadata.ch_xyz1 is not None:
            x1, y1 = [float(d) for d in self.metadata.ch_xyz1.split(":")]
            x2, y2 = [float(d) for d in self.metadata.ch_xyz2.split(":")]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 100.0
            length = np.round(length, 2)
        return length

    @property
    def azimuth(self):
        """
        azimuth of instrument setup
        """
        if self.metadata.ch_azimuth is not None:
            return float(self.metadata.ch_azimuth)
        elif self.metadata.rx_xazimuth is not None:
            return float(self.metadata.rx_xazimuth)
        else:
            return None

    @property
    def component(self):
        """
        channel
        """
        return self.metadata.ch_cmp.lower()

    @property
    def latitude(self):
        """
        latitude in decimal degrees
        """
        return self.header.lat

    @property
    def longitude(self):
        """
        longitude in decimal degrees
        """
        return self.header.long

    @property
    def elevation(self):
        """
        elevation in meters
        """
        return self.header.alt

    @property
    def sample_rate(self):
        """
        sampling rate
        """
        return self.header.ad_rate

    @sample_rate.setter
    def sample_rate(self, sampling_rate):
        """
        sampling rate
        """
        if sampling_rate is not None:
            self.header.ad_rate = float(sampling_rate)

    @property
    def start(self):
        if self.gps_stamps is not None:
            return self.get_UTC_date_time(
                self.header.gpsweek, self.gps_stamps["time"][0]
            )
        return self.zen_schedule

    @property
    def end(self):
        if self.gps_stamps is not None:
            return self.get_UTC_date_time(
                self.header.gpsweek, self.gps_stamps["time"][-1]
            )
        return self.start + (self.n_samples / self.sample_rate)

    @property
    def zen_schedule(self):
        """
        zen schedule data and time
        """

        if self.header.old_version is True:
            return MTime(time_stamp=self.header.schedule)
        return self.schedule.initial_start

    @zen_schedule.setter
    def zen_schedule(self, schedule_dt):
        """
        on setting set schedule datetime
        """
        if not isinstance(schedule_dt, MTime):
            schedule_dt = MTime(time_stamp=schedule_dt)
            raise TypeError("New schedule datetime must be type datetime.datetime")
        self.schedule.initial_start = schedule_dt

    @property
    def coil_number(self):
        """
        coil number
        """
        if self.metadata.cal_ant is not None:
            return self.metadata.cal_ant
        elif self.metadata.ch_number is not None:
            return self.metadata.ch_number
        else:
            return None

    @property
    def channel_number(self):
        if self.metadata.ch_number:
            ch_num = int(float(self.metadata.ch_number))
            if ch_num > 6:
                try:
                    ch_num = self.ch_dict[self.component]
                except KeyError:
                    ch_num = 6
            return ch_num
        else:
            try:
                return self.ch_dict[self.component]
            except KeyError:
                return 0

    @property
    def channel_metadata(self):
        """Channel metadata"""

        # fill the time series object
        if "e" in self.component:
            ch = Electric()
            ch.dipole_length = self.dipole_length
            ch.ac.start = (
                self.time_series[0 : int(self.sample_rate)].std()
                * self.header.ch_factor
            )
            ch.ac.end = (
                self.time_series[-int(self.sample_rate) :].std() * self.header.ch_factor
            )
            ch.dc.start = (
                self.time_series[0 : int(self.sample_rate)].mean()
                * self.header.ch_factor
            )
            ch.dc.end = (
                self.time_series[-int(self.sample_rate) :].mean()
                * self.header.ch_factor
            )
        elif "h" in self.component:
            ch = Magnetic()
            ch.sensor.id = self.coil_number
            ch.sensor.manufacturer = "Geotell"
            ch.sensor.model = "ANT-4"
            ch.sensor.type = "induction coil"
            ch.h_field_max.start = (
                self.time_series[0 : int(self.sample_rate)].max()
                * self.header.ch_factor
            )
            ch.h_field_max.end = (
                self.time_series[-int(self.sample_rate) :].max() * self.header.ch_factor
            )
            ch.h_field_min.start = (
                self.time_series[0 : int(self.sample_rate)].min()
                * self.header.ch_factor
            )
            ch.h_field_min.end = (
                self.time_series[-int(self.sample_rate) :].min() * self.header.ch_factor
            )
        ch.time_period.start = self.start.isoformat()
        ch.time_period.end = self.end.isoformat()
        ch.component = self.component
        ch.sample_rate = self.sample_rate
        ch.measurement_azimuth = self.azimuth
        if ch.component in ["ey", "e2"] and self.azimuth == 0:
            ch.measurement_azimuth = 90
        ch.units = "digital counts"
        ch.channel_number = self.channel_number
        ch.filter.name = self.channel_response.names
        ch.filter.applied = [True] * len(self.channel_response.names)

        return ch

    @property
    def station_metadata(self):
        """station metadta"""

        sm = Station()
        sm.id = self.station
        sm.fdsn.id = self.station
        sm.location.latitude = self.latitude
        sm.location.longitude = self.longitude
        sm.location.elevation = self.elevation
        sm.time_period.start = self.start.isoformat()
        sm.time_period.end = self.end.isoformat()
        sm.acquired_by.author = self.metadata.gdp_operator

        return sm

    @property
    def run_metadata(self):
        """Run metadata"""
        rm = Run()
        rm.data_logger.firmware.version = self.header.version
        rm.data_logger.id = self.header.data_logger
        rm.data_logger.manufacturer = "Zonge International"
        rm.data_logger.model = "ZEN"
        rm.time_period.start = self.start.isoformat()
        rm.time_period.end = self.end.isoformat()
        rm.sample_rate = self.sample_rate
        rm.data_type = "MTBB"
        rm.time_period.start = self.start.isoformat()
        rm.time_period.end = self.end.isoformat()
        rm.acquired_by.author = self.metadata.gdp_operator
        rm.id = f"sr{int(self.sample_rate)}_001"

        return rm

    @property
    def counts2mv_filter(self):
        """
        Create a counts2mv coefficient filter

        .. note:: Needs to be 1/channel factor because we divided the
         instrument response from the data.

        """

        c2mv = CoefficientFilter()
        c2mv.units_in = "millivolts"
        c2mv.units_out = "digital counts"
        c2mv.name = "zen_counts2mv"
        c2mv.gain = 1.0 / self.header.ch_factor
        c2mv.comments = "digital counts to millivolts"

        return c2mv

    @property
    def coil_response(self):
        """
        Make the coile response into a FAP filter

        Phase must be in radians
        """
        fap = None
        # if there is no calibration file get from Z3D file, though it seems
        # like these are not read in properly.
        if self.calibration_fn in [None, "None"]:
            # looks like zen outputs radial frequency
            if self.metadata.cal_ant is not None:
                fap = FrequencyResponseTableFilter()
                fap.units_in = "nanotesla"
                fap.units_out = "millivolts"
                fap.frequencies = (1 / (2 * np.pi)) * self.metadata.coil_cal.frequency
                fap.amplitudes = self.metadata.coil_cal.amplitude
                fap.phases = self.metadata.coil_cal.phase / 1e3
                fap.name = f"ant4_{self.coil_number}_response"
                fap.comments = "induction coil response read from z3d file"
        else:
            c = CoilResponse(self.calibration_fn)
            if c.has_coil_number(self.coil_number):
                fap = c.get_coil_response_fap(self.coil_number)
        return fap

    @property
    def zen_response(self):
        """
        Zen response, not sure the full calibration comes directly from the
        Z3D file, so skipping for now.  Will have to read a Zen##.cal file
        to get the full calibration.  This shouldn't be a big issue cause it
        should roughly be the same for all channels and since the TF is
        computing the ratio they will cancel out.  Though we should look
        more into this if just looking at calibrate time series.

        """
        return None
        fap = None
        find = False
        return
        if self.metadata.board_cal not in [None, []]:
            if self.metadata.board_cal[0][0] == "":
                return fap
            sr_dict = {256: 0, 1024: 1, 4096: 4}
            sr_int = sr_dict[int(self.sample_rate)]
            fap_table = self.metadata.board_cal[
                np.where(self.metadata.board_cal.rate == sr_int)
            ]
            frequency = fap_table.frequency
            amplitude = fap_table.amplitude
            phase = fap_table.phase / 1e3
            find = True
        elif self.metadata.cal_board is not None:
            try:
                fap_dict = self.metadata.cal_board[int(self.sample_rate)]
                frequency = fap_dict["frequency"]
                amplitude = fap_dict["amplitude"]
                phase = fap_dict["phase"]
                find = True
            except KeyError:
                try:
                    fap_str = self.metadata.cal_board["cal.ch"]
                    for ss in fap_str.split(";"):
                        freq, _, resp = ss.split(",")
                        ff, amp, phs = [float(item) for item in resp.split(":")]
                        if float(freq) == self.sample_rate:
                            frequency = ff
                            amplitude = amp
                            phase = phs / 1e3
                    find = True
                except KeyError:
                    return fap
        if find:
            freq = np.logspace(np.log10(6.00000e-04), np.log10(8.19200e03), 48)
            amp = np.ones(48)
            phases = np.zeros(48)
            for item_f, item_a, item_p in zip(frequency, amplitude, phase):
                index = np.abs(freq - item_f).argmin()
                freq[index] = item_f
                amp[index] = item_a
                phases[index] = item_p
            fap = FrequencyResponseTableFilter()
            fap.units_in = "millivolts"
            fap.units_out = "millivolts"
            fap.frequencies = freq
            fap.amplitudes = amp
            fap.phases = phases
            fap.name = (
                f"{self.header.data_logger.lower()}_{self.sample_rate:.0f}_response"
            )
            fap.comments = "data logger response read from z3d file"
            return fap
        return None

    @property
    def channel_response(self):
        filter_list = []
        # don't have a good handle on the zen response yet.
        # if self.zen_response:
        #     filter_list.append(self.zen_response)
        if self.coil_response:
            filter_list.append(self.coil_response)
        elif self.dipole_filter:
            filter_list.append(self.dipole_filter)

        filter_list.append(self.counts2mv_filter)
        return ChannelResponse(filters_list=filter_list)

    @property
    def dipole_filter(self):
        dipole = None
        # needs to be the inverse for processing
        if self.dipole_length != 0:
            dipole = CoefficientFilter()
            dipole.units_in = "millivolts per kilometer"
            dipole.units_out = "millivolts"
            dipole.name = f"dipole_{self.dipole_length:.2f}m"
            dipole.gain = self.dipole_length / 1000.0
            dipole.comments = "convert to electric field"
        return dipole

    def _get_gps_stamp_type(self, old_version=False):
        """
        get the correct stamp type.
        Older versions the stamp length was 36 bits
        New versions have a 64 bit stamp
        """

        if old_version is True:
            self._gps_dtype = np.dtype(
                [
                    ("gps", np.int32),
                    ("time", np.int32),
                    ("lat", np.float64),
                    ("lon", np.float64),
                    ("block_len", np.int32),
                    ("gps_accuracy", np.int32),
                    ("temperature", np.float32),
                ]
            )
            self._gps_stamp_length = 36
            self._gps_bytes = self._gps_stamp_length / 4
            self._gps_flag_0 = -1
            self._block_len = int(self._gps_stamp_length + self.sample_rate * 4)
            self.gps_flag = self._gps_f0
        else:
            return

    # ======================================
    def _read_header(self, fn=None, fid=None):
        """
        read header information from Z3D file
        Arguments
        ---------------
            **fn** : string
                     full path to Z3D file to read
            **fid** : file object
                      if the file is open give the file id object
        Outputs:
        ----------
            * fills the Zen3ZD.header object's attributes

        Example with just a file name
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_header(fn)

        Example with file object
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dfid = open(fn, 'rb')
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_header(fid=Z3Dfid)
        """

        if fn is not None:
            self.fn = fn
        self.header.read_header(fn=self.fn, fid=fid)
        if self.header.old_version:
            if self.header.box_number is None:
                self.header.box_number = "6666"

    # ======================================
    def _read_schedule(self, fn=None, fid=None):
        """
        read schedule information from Z3D file
        Arguments
        ---------------
            **fn** : string
                     full path to Z3D file to read
            **fid** : file object
                      if the file is open give the file id object
        Outputs:
        ----------
            * fills the Zen3ZD.schedule object's attributes

        Example with just a file name
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_schedule(fn)

        Example with file object
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dfid = open(fn, 'rb')
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_schedule(fid=Z3Dfid)
        """

        if fn is not None:
            self.fn = fn
        self.schedule.read_schedule(fn=self.fn, fid=fid)
        if self.header.old_version:
            self.schedule.initial_start = MTime(
                time_stamp=self.header.schedule, gps_time=True
            )

    # ======================================
    def _read_metadata(self, fn=None, fid=None):
        """
        read header information from Z3D file
        Arguments
        ---------------
            **fn** : string
                     full path to Z3D file to read
            **fid** : file object
                      if the file is open give the file id object
        Outputs:
        ----------
            * fills the Zen3ZD.metadata object's attributes

        Example with just a file name
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_metadata(fn)

        Example with file object
        ------------
            >>> import mtpy.usgs.zen as zen
            >>> fn = r"/home/mt/mt01/mt01_20150522_080000_256_EX.Z3D"
            >>> Z3Dfid = open(fn, 'rb')
            >>> Z3Dobj = zen.Zen3D()
            >>> Z3Dobj.read_metadata(fid=Z3Dfid)
        """

        if fn is not None:
            self.fn = fn
        if self.header.old_version:
            self.metadata._schedule_metadata_len = 0
        self.metadata.read_metadata(fn=self.fn, fid=fid)

    # =====================================
    def read_all_info(self):
        """
        Read header, schedule, and metadata
        """
        with open(self.fn, "rb") as file_id:
            self._read_header(fid=file_id)
            self._read_schedule(fid=file_id)
            self._read_metadata(fid=file_id)

    def _find_first_gps_flag(self, fid) -> int:
        """
        find the first GPS flag, shoud be at the end of the metadata, but sometimes
        that is incorrect.  There is a few extra bytes of data.  So need to go
        byte by byte to find the first GPS flag.

        """
        find_gps_flag = False
        fid_tell = self.metadata.m_tell - 1
        fid.seek(fid_tell)
        # adding a fail safe to not have an infinite loop
        while not find_gps_flag or fid_tell < 15000:
            fid_tell += 1
            fid.seek(fid_tell)
            line = fid.read(4)
            try:
                line = np.frombuffer(line, np.int32)[0]
                if line == np.int32(2147483647):
                    return fid_tell
            except AttributeError:
                continue

    def _read_raw_string(self, fid):
        """
        read raw sting into data

        :param fid: DESCRIPTION
        :type fid: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        # move the read value to where the end of the metadata is
        self.metadata.m_tell = self._find_first_gps_flag(fid)
        fid.seek(self.metadata.m_tell)

        # initalize a data array filled with zeros, everything goes into
        # this array then we parse later
        data = np.zeros(self.n_samples, dtype=np.int32)
        # go over a while loop until the data cound exceed the file size
        data_count = 0
        while True:
            # need to make sure the last block read is a multiple of 32 bit
            read_len = min(
                [
                    self._block_len,
                    int(32 * ((self.file_size - fid.tell()) // 32)),
                ]
            )
            test_str = np.frombuffer(fid.read(read_len), dtype=np.int32)
            if len(test_str) == 0:
                break
            data[data_count : data_count + len(test_str)] = test_str
            data_count += test_str.size
        return data

    def _unpack_data(self, data, gps_stamp_index):
        """ """

        for ii, gps_find in enumerate(gps_stamp_index):
            try:
                data[gps_find + 1]
            except IndexError:
                self.logger.warning(
                    f"Failed gps stamp {ii+1} out of {len(gps_stamp_index)}"
                )
                break
            if (
                self.header.old_version is True
                or data[gps_find + 1] == self._gps_flag_1
            ):
                gps_str = struct.pack(
                    "<" + "i" * int(self._gps_bytes),
                    *data[int(gps_find) : int(gps_find + self._gps_bytes)],
                )
                self.gps_stamps[ii] = np.frombuffer(gps_str, dtype=self._gps_dtype)
                if ii > 0:
                    self.gps_stamps[ii]["block_len"] = (
                        gps_find - gps_stamp_index[ii - 1] - self._gps_bytes
                    )
                elif ii == 0:
                    self.gps_stamps[ii]["block_len"] = 0
                data[int(gps_find) : int(gps_find + self._gps_bytes)] = 0
        return data

    # ======================================
    def read_z3d(self, z3d_fn=None):
        """
        read in z3d file and populate attributes accordingly

        1. Read in the entire file as chunks as np.int32.

        2. Extract the gps stamps and convert accordingly. Check to make sure
           gps time stamps are 1 second apart and incrementing as well as
           checking the number of data points between stamps is the
           same as the sampling rate.

        3. Converts gps_stamps['time'] to seconds relative to header.gps_week
            Note we skip the first two gps stamps because there is something
            wrong with the data there due to some type of buffering.
            Therefore the first GPS time is when the time series starts, so you
            will notice that gps_stamps[0]['block_len'] = 0, this is because there
            is nothing previous to this time stamp and so the 'block_len' measures
            backwards from the corresponding time index.

        4. Put the data chunks into Pandas data frame that is indexed by time

        :Example:

        >>> from mth5.io import zen
        >>> z_obj = zen.Z3D(r"home/mt_data/zen/mt001.z3d")
        >>> z_obj.read_z3d()

        """

        if z3d_fn is not None:
            self.fn = z3d_fn
        self.logger.debug(f"Reading {self.fn}")
        st = datetime.datetime.now()

        # using the with statement works in Python versions 2.7 or higher
        # the added benefit of the with statement is that it will close the
        # file object upon reading completion.
        with open(self.fn, "rb") as file_id:
            self._read_header(fid=file_id)
            self._read_schedule(fid=file_id)
            self._read_metadata(fid=file_id)

            if self.header.old_version is True:
                self._get_gps_stamp_type(True)
            data = self._read_raw_string(file_id)
        self.raw_data = data.copy()

        # find the gps stamps
        gps_stamp_find = self.get_gps_stamp_index(data, self.header.old_version)

        # skip the first two stamps and trim data
        try:
            data = data[gps_stamp_find[self.num_sec_to_skip] :]
        except IndexError:
            msg = f"Data is too short, cannot open file {self.fn}"
            self.logger.error(msg)
            raise ZenGPSError(msg)
        # find gps stamps of the trimmed data
        gps_stamp_find = self.get_gps_stamp_index(data, self.header.old_version)

        # read data chunks and GPS stamps
        self.gps_stamps = np.zeros(len(gps_stamp_find), dtype=self._gps_dtype)
        data = self._unpack_data(data, gps_stamp_find)

        # fill the time series
        self.time_series = data[np.nonzero(data)]

        # validate everything
        if not self.validate_gps_time():
            self.logger.warning(
                f"GPS stamps are not 1 second apart for file {self.fn.name}."
            )
        if not self.validate_time_blocks():
            self.logger.warning(
                f"Time block between stamps was not the sample rate for file {self.fn.name}"
            )
        self.convert_gps_time()
        self.zen_schedule = self.check_start_time()
        self.logger.debug(f"found {self.gps_stamps.shape[0]} GPS time stamps")
        self.logger.debug(f"found {self.time_series.size} data points")

        # time it
        et = datetime.datetime.now()
        read_time = (et - st).total_seconds()
        self.logger.debug(f"Reading data took: {read_time:.3f} seconds")

    # =================================================
    def get_gps_stamp_index(self, ts_data, old_version=False):
        """
        locate the time stamps in a given time series.

        Looks for gps_flag_0 first, if the file is newer, then makes sure the
        next value is gps_flag_1

        :returns: list of gps stamps indicies
        """

        # find the gps stamps
        gps_stamp_find = np.where(ts_data == self._gps_flag_0)[0]

        if old_version is False:
            gps_stamp_find = [
                gps_find
                for gps_find in gps_stamp_find
                if ts_data[gps_find + 1] == self._gps_flag_1
            ]
        return gps_stamp_find

    # =================================================
    def trim_data(self):
        """
        apparently need to skip the first 2 seconds of data because of
        something to do with the SD buffer

        This method will be deprecated after field testing

        """
        # the block length is the number of data points before the time stamp
        # therefore the first block length is 0.  The indexing in python
        # goes to the last index - 1 so we need to put in 3
        ts_skip = self.gps_stamps["block_len"][0:3].sum()
        self.gps_stamps = self.gps_stamps[2:]
        self.gps_stamps[0]["block_len"] = 0
        self.time_series = self.time_series[ts_skip:]

    # =================================================
    def check_start_time(self):
        """
        check to make sure the scheduled start time is similar to
        the first good gps stamp
        """

        # make sure the time is in gps time
        zen_start_utc = self.get_UTC_date_time(
            self.header.gpsweek, self.gps_stamps["time"][0]
        )

        # estimate the time difference between the two
        time_diff = zen_start_utc - self.schedule.initial_start
        if time_diff > self._max_time_diff:
            self.logger.warning(f"ZEN Scheduled time was {self.schedule.initial_start}")
            self.logger.warning(f"1st good stamp was {zen_start_utc}")
            self.logger.warning(f"difference of {time_diff:.2f} seconds")

        self.logger.debug(f"Scheduled time was {self.schedule.initial_start}")
        self.logger.debug(f"1st good stamp was {zen_start_utc}")
        self.logger.debug(f"difference of {time_diff:.2f} seconds")

        return zen_start_utc

    # ==================================================
    def validate_gps_time(self):
        """
        make sure each time stamp is 1 second apart

        """
        # need to put the gps time into seconds
        t_diff = np.diff(self.gps_stamps["time"]) / 1024

        bad_times = np.where(abs(t_diff) > 1)[0]
        if len(bad_times) > 0:
            for bb in bad_times:
                self.logger.debug(
                    f"ZEN bad GPS time at index {bb} > 1 second " f"({t_diff[bb]})"
                )
            return False
        return True

    # ===================================================
    def validate_time_blocks(self):
        """
        validate gps time stamps and make sure each block is the proper length

        """
        # first check if the gps stamp blocks are of the correct length
        bad_blocks = np.where(self.gps_stamps["block_len"][1:] != self.header.ad_rate)[
            0
        ]

        if len(bad_blocks) > 0:
            if bad_blocks.max() < 5:
                ts_skip = self.gps_stamps["block_len"][0 : bad_blocks[-1] + 1].sum()
                self.gps_stamps = self.gps_stamps[bad_blocks[-1] :]
                self.time_series = self.time_series[ts_skip:]

                self.logger.warning(f"Skipped the first {bad_blocks[-1]} seconds")
                self.logger.warning(f"Skipped first {ts_skip} poins in time series")
            return False
        return True

    # ==================================================
    def convert_gps_time(self):
        """
        convert gps time integer to relative seconds from gps_week

        """
        # need to convert gps_time to type float from int
        dt = self._gps_dtype.descr
        if self.header.old_version is True:
            dt[1] = ("time", np.float32)
        else:
            dt[2] = ("time", np.float32)
        self.gps_stamps = self.gps_stamps.astype(np.dtype(dt))

        # convert to seconds
        # these are seconds relative to the gps week
        time_conv = self.gps_stamps["time"].copy() / 1024.0
        time_ms = (time_conv - np.floor(time_conv)) * 1.024
        time_conv = np.floor(time_conv) + time_ms

        self.gps_stamps["time"][:] = time_conv

    # ==================================================
    def convert_counts_to_mv(self, data):
        """
        convert the time series from counts to millivolts

        """

        data *= self._counts_to_mv_conversion
        return data

    # ==================================================
    def convert_mv_to_counts(self, data):
        """
        convert millivolts to counts assuming no other scaling has been applied

        """

        data /= self._counts_to_mv_conversion
        return data

    # ==================================================
    def get_gps_time(self, gps_int, gps_week=0):
        """
        from the gps integer get the time in seconds.

        :param int gps_int: integer from the gps time stamp line
        :param int gps_week: relative gps week, if the number of seconds is
                            larger than a week then a week is subtracted from
                            the seconds and computed from gps_week += 1
        :returns: gps_time as number of seconds from the beginning of the relative
                  gps week.

        """

        gps_seconds = gps_int / 1024.0

        gps_ms = (gps_seconds - np.floor(gps_int / 1024.0)) * (1.024)

        cc = 0
        if gps_seconds > self._week_len:
            gps_week += 1
            cc = gps_week * self._week_len
            gps_seconds -= self._week_len
        gps_time = np.floor(gps_seconds) + gps_ms + cc

        return gps_time, gps_week

    # ==================================================
    def get_UTC_date_time(self, gps_week, gps_time):
        """
        get the actual date and time of measurement as UTC.


        :param int gps_week: integer value of gps_week that the data was collected
        :param int gps_time: number of seconds from beginning of gps_week

        :return: :class:`mth5.utils.mttime.MTime`

        """
        # need to check to see if the time in seconds is more than a gps week
        # if it is add 1 to the gps week and reduce the gps time by a week
        if gps_time > self._week_len:
            gps_week += 1
            gps_time -= self._week_len
        # compute seconds using weeks and gps time
        utc_seconds = (
            self._gps_epoch.epoch_seconds + (gps_week * self._week_len) + gps_time
        )

        # compute date and time from seconds and return a datetime object
        # easier to manipulate later, must be in nanoseconds
        return MTime(time_stamp=utc_seconds, gps_time=True)

    # =================================================
    def to_channelts(self):
        """
        fill time series object
        """

        return ChannelTS(
            self.channel_metadata.type,
            data=self.time_series,
            channel_metadata=self.channel_metadata,
            station_metadata=self.station_metadata,
            run_metadata=self.run_metadata,
            channel_response=self.channel_response,
        )


# ==============================================================================
#  Error instances for Zen
# ==============================================================================
class ZenGPSError(Exception):
    """
    error for gps timing
    """


class ZenSamplingRateError(Exception):
    """
    error for different sampling rates
    """


class ZenInputFileError(Exception):
    """
    error for input files
    """


def read_z3d(fn, calibration_fn=None, logger_file_handler=None):
    """
    generic tool to read z3d file
    """

    z3d_obj = Z3D(fn, calibration_fn=calibration_fn)
    if logger_file_handler:
        z3d_obj.logger.addHandler(logger_file_handler)
    try:
        z3d_obj.read_z3d()
    except ZenGPSError as error:
        z3d_obj.logger.exception(error)
        z3d_obj.logger.warning(f"Skipping {fn}, check file for GPS timing.")
        return None
    return z3d_obj.to_channelts()
