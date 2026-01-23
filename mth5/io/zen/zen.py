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

from __future__ import annotations

import datetime
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from loguru import logger
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries import AppliedFilter, Electric, Magnetic, Run, Station
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
    A class for reading and processing Z3D files output by Zen data loggers.

    This class handles the parsing of Z3D binary files which contain GPS-stamped
    time series data from magnetotelluric measurements. It provides methods for
    reading file headers, metadata, schedule information, and time series data,
    as well as converting between different units and formats.

    Parameters
    ----------
    fn : str or Path, optional
        Full path to the .Z3D file to be read. Default is None.
    **kwargs : dict
        Additional keyword arguments including:
        - stamp_len : int, default 64
            GPS stamp length in bits

    Attributes
    ----------
    fn : Path or None
        Path to the Z3D file
    calibration_fn : str or None
        Path to calibration file
    header : Z3DHeader
        Header information object
    schedule : Z3DSchedule
        Schedule information object
    metadata : Z3DMetadata
        Metadata information object
    gps_stamps : numpy.ndarray or None
        Array of GPS time stamps
    time_series : numpy.ndarray or None
        Time series data array
    sample_rate : float or None
        Data sampling rate in Hz
    units : str
        Data units, default 'counts'

    Notes
    -----
    GPS data type is formatted as::

        numpy.dtype([('flag0', numpy.int32),
                     ('flag1', numpy.int32),
                     ('time', numpy.int32),
                     ('lat', numpy.float64),
                     ('lon', numpy.float64),
                     ('num_sat', numpy.int32),
                     ('gps_sens', numpy.int32),
                     ('temperature', numpy.float32),
                     ('voltage', numpy.float32),
                     ('num_fpga', numpy.int32),
                     ('num_adc', numpy.int32),
                     ('pps_count', numpy.int32),
                     ('dac_tune', numpy.int32),
                     ('block_len', numpy.int32)])

    Examples
    --------
    >>> from mth5.io.zen import Z3D
    >>> z3d = Z3D(r"/path/to/data/station_20150522_080000_256_EX.Z3D")
    >>> z3d.read_z3d()
    >>> print(f"Found {z3d.gps_stamps.shape[0]} GPS time stamps")
    >>> print(f"Found {z3d.time_series.size} data points")
    """

    def __init__(self, fn: str | Path | None = None, **kwargs: Any) -> None:
        """
        Initialize Z3D file reader object.

        Parameters
        ----------
        fn : str or Path, optional
            Full path to the Z3D file to be processed, by default None
        **kwargs : dict
            Additional keyword arguments:
            - stamp_len : int, default 64
                GPS stamp length in bits

        Examples
        --------
        >>> z3d = Z3D("/path/to/file.Z3D")
        >>> z3d.read_z3d()
        >>> print(z3d.station)
        """
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
        self.num_sec_to_skip = 1

        self.units = "digital counts"
        self.sample_rate = None
        self.time_series = None
        self._max_time_diff = 20

        self.ch_dict = {"hx": 1, "hy": 2, "hz": 3, "ex": 4, "ey": 5}

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def fn(self) -> Path | None:
        """
        Get the Z3D file path.

        Returns
        -------
        Path or None
            Path to the Z3D file, or None if not set.
        """
        return self._fn

    @fn.setter
    def fn(self, fn: str | Path | None) -> None:
        """
        Set the Z3D file path.

        Parameters
        ----------
        fn : str, Path, or None
            Path to the Z3D file to set.
        """
        if fn is not None:
            self._fn = Path(fn)
        else:
            self._fn = None

    @property
    def file_size(self) -> int:
        """
        Get the size of the Z3D file in bytes.

        Returns
        -------
        int
            File size in bytes, or 0 if no file is set.
        """
        if self.fn is not None:
            return self.fn.stat().st_size
        return 0

    @property
    def n_samples(self) -> int:
        """
        Get the number of data samples in the file.

        Returns
        -------
        int
            Number of data samples. Calculated from file size if time_series
            is not loaded, otherwise returns the actual array size.

        Notes
        -----
        Calculation assumes 4 bytes per sample and accounts for metadata blocks.
        If sample_rate is available, adds buffer for GPS stamps.
        """
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
    def station(self) -> str | None:
        """
        Get the station name.

        Returns
        -------
        str or None
            Station identifier name.
        """
        return self.metadata.station

    @station.setter
    def station(self, station: str) -> None:
        """
        Set the station name.

        Parameters
        ----------
        station : str
            Station identifier name to set.
        """
        self.metadata.station = station

    @property
    def dipole_length(self) -> float:
        """
        Get the dipole length for electric field measurements.

        Returns
        -------
        float
            Dipole length in meters. Calculated from electrode positions
            if not directly specified in metadata. Returns 0 for magnetic
            channels or if positions are not available.

        Notes
        -----
        Length is calculated from xyz coordinates using Euclidean distance
        formula when position data is available in metadata.
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
    def azimuth(self) -> float | None:
        """
        Get the azimuth of instrument setup.

        Returns
        -------
        float or None
            Azimuth angle in degrees from north, or None if not available.
        """
        if self.metadata.ch_azimuth is not None:
            return float(self.metadata.ch_azimuth)
        elif self.metadata.rx_xazimuth is not None:
            return float(self.metadata.rx_xazimuth)
        else:
            return None

    @property
    def component(self) -> str:
        """
        Get the channel component identifier.

        Returns
        -------
        str
            Channel component name in lowercase (e.g., 'ex', 'hy', 'hz').
        """
        return self.metadata.ch_cmp.lower()

    @property
    def latitude(self) -> float | None:
        """
        Get the latitude in decimal degrees.

        Returns
        -------
        float or None
            Latitude coordinate in decimal degrees, or None if not available.
        """
        return self.header.lat

    @property
    def longitude(self) -> float | None:
        """
        Get the longitude in decimal degrees.

        Returns
        -------
        float or None
            Longitude coordinate in decimal degrees, or None if not available.
        """
        return self.header.long

    @property
    def elevation(self) -> float | None:
        """
        Get the elevation in meters.

        Returns
        -------
        float or None
            Elevation above sea level in meters, or None if not available.
        """
        return self.header.alt

    @property
    def sample_rate(self) -> float | None:
        """
        Get the sampling rate in Hz.

        Returns
        -------
        float or None
            Data sampling rate in samples per second, or None if not available.
        """
        return self.header.ad_rate

    @sample_rate.setter
    def sample_rate(self, sampling_rate: float | None) -> None:
        """
        Set the sampling rate.

        Parameters
        ----------
        sampling_rate : float or None
            Sampling rate in Hz to set.
        """
        if sampling_rate is not None:
            self.header.ad_rate = float(sampling_rate)

    @property
    def start(self) -> MTime:
        """
        Get the start time of the data.

        Returns
        -------
        MTime
            Start time from GPS stamps if available, otherwise scheduled time.
        """
        if self.gps_stamps is not None:
            return self.get_UTC_date_time(
                self.header.gpsweek, float(self.gps_stamps["time"][0])
            )
        return self.zen_schedule

    @property
    def end(self) -> MTime | float:
        """
        Get the end time of the data.

        Returns
        -------
        MTime or float
            End time from GPS stamps if available, otherwise calculated
            from start time and number of samples.
        """
        if self.gps_stamps is not None:
            return self.get_UTC_date_time(
                self.header.gpsweek, float(self.gps_stamps["time"][-1])
            )
        return self.start + (self.n_samples / self.sample_rate)

    @property
    def zen_schedule(self) -> MTime:
        """
        Get the zen schedule date and time.

        Returns
        -------
        MTime
            Scheduled start time from header or schedule object.
        """
        if self.header.old_version is True:
            return MTime(time_stamp=self.header.schedule)
        return self.schedule.initial_start

    @zen_schedule.setter
    def zen_schedule(self, schedule_dt: MTime | str | datetime.datetime) -> None:
        """
        Set the zen schedule datetime.

        Parameters
        ----------
        schedule_dt : MTime, str, or datetime.datetime
            Schedule datetime to set.

        Raises
        ------
        TypeError
            If schedule_dt is not a valid time type.
        """
        if not isinstance(schedule_dt, MTime):
            schedule_dt = MTime(time_stamp=schedule_dt)
            raise TypeError("New schedule datetime must be type datetime.datetime")
        self.schedule.initial_start = schedule_dt

    @property
    def coil_number(self) -> str | None:
        """
        Get the coil number identifier.

        Returns
        -------
        str or None
            Coil antenna number identifier, or None if not available.
        """
        if self.metadata.cal_ant is not None:
            return self.metadata.cal_ant
        elif self.metadata.ch_number is not None:
            return self.metadata.ch_number
        else:
            return None

    @property
    def channel_number(self) -> int:
        """
        Get the channel number.

        Returns
        -------
        int
            Channel number identifier. Maps component names to standard
            channel numbers or uses metadata channel number.
        """
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
        """
        Generate Channel metadata object from Z3D file information.

        Creates either an Electric or Magnetic metadata object based on the
        component type, populated with channel-specific parameters, sensor
        information, and data statistics.

        Returns
        -------
        Electric or Magnetic
            Channel metadata object appropriate for the measurement type:
            - Electric: includes dipole length, AC/DC statistics
            - Magnetic: includes sensor details, field min/max values

        Notes
        -----
        Electric channels (ex, ey) get dipole length and voltage statistics.
        Magnetic channels (hx, hy, hz) get sensor information and field
        strength statistics computed from the first and last seconds of data.

        Examples
        --------
        >>> z3d = Z3D("/path/to/file.Z3D")
        >>> z3d.read_z3d()
        >>> ch_meta = z3d.channel_metadata
        >>> print(f"Channel component: {ch_meta.component}")
        """

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
        for count, f in enumerate(self.channel_response.filters_list, 1):
            ch.add_filter(AppliedFilter(name=f.name, applied=True, stage=count))
        return ch

    @property
    def station_metadata(self):
        """
        Generate Station metadata object from Z3D file information.

        Creates a Station metadata object populated with location and timing
        information extracted from the Z3D file header and metadata.

        Returns
        -------
        Station
            Station metadata object with populated fields including station ID,
            coordinates, elevation, time period, and operator information.

        Examples
        --------
        >>> z3d = Z3D("/path/to/file.Z3D")
        >>> z3d.read_all_info()
        >>> station_meta = z3d.station_metadata
        >>> print(station_meta.id)
        """

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
        """
        Generate Run metadata object from Z3D file information.

        Creates a Run metadata object populated with data logger information,
        timing details, and measurement parameters extracted from the Z3D file.

        Returns
        -------
        Run
            Run metadata object with populated fields including data logger
            details, sample rate, time period, and data type information.

        Examples
        --------
        >>> z3d = Z3D("/path/to/file.Z3D")
        >>> z3d.read_all_info()
        >>> run_meta = z3d.run_metadata
        >>> print(f"Sample rate: {run_meta.sample_rate}")
        """
        rm = Run()
        rm.data_logger.firmware.version = self.header.version
        rm.data_logger.id = self.header.data_logger
        rm.data_logger.manufacturer = "Zonge International"
        rm.data_logger.model = "ZEN"
        rm.time_period.start = self.start.isoformat()
        rm.time_period.end = self.end.isoformat()
        rm.sample_rate = self.sample_rate
        rm.data_type = "BBMT"
        rm.time_period.start = self.start.isoformat()
        rm.time_period.end = self.end.isoformat()
        rm.acquired_by.author = self.metadata.gdp_operator
        rm.id = f"sr{int(self.sample_rate)}_001"

        return rm

    @property
    def counts2mv_filter(self):
        """
        Create a counts to milliVolt coefficient filter.

        Generate a coefficient filter for converting digital counts to milliVolt
        using the channel factor from the Z3D file header.

        Returns
        -------
        CoefficientFilter
            Filter object configured for counts to milliVolt conversion with
            gain set to the inverse of the channel factor.

        Notes
        -----
        The gain is set to 1/channel_factor because this represents the
        inverse operation when the instrument response has been divided
        from the data during processing.

        Examples
        --------
        >>> z3d = Z3D("/path/to/file.Z3D")
        >>> z3d.read_all_info()
        >>> filter_obj = z3d.counts2mv_filter
        >>> print(f"Conversion gain: {filter_obj.gain}")
        """

        c2mv = CoefficientFilter()
        c2mv.units_in = "milliVolt"
        c2mv.units_out = "digital counts"
        c2mv.name = "zen_counts2mv"
        c2mv.gain = 1.0 / self.header.ch_factor
        c2mv.comments = "digital counts to milliVolt"

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
                fap.units_in = "nanoTesla"
                fap.units_out = "milliVolt"
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
        # fap = None
        # find = False
        # return
        # if self.metadata.board_cal not in [None, []]:
        #     if self.metadata.board_cal[0][0] == "":
        #         return fap
        #     sr_dict = {256: 0, 1024: 1, 4096: 4}
        #     sr_int = sr_dict[int(self.sample_rate)]
        #     fap_table = self.metadata.board_cal[
        #         np.where(self.metadata.board_cal.rate == sr_int)
        #     ]
        #     frequency = fap_table.frequency
        #     amplitude = fap_table.amplitude
        #     phase = fap_table.phase / 1e3
        #     find = True
        # elif self.metadata.cal_board is not None:
        #     try:
        #         fap_dict = self.metadata.cal_board[int(self.sample_rate)]
        #         frequency = fap_dict["frequency"]
        #         amplitude = fap_dict["amplitude"]
        #         phase = fap_dict["phase"]
        #         find = True
        #     except KeyError:
        #         try:
        #             fap_str = self.metadata.cal_board["cal.ch"]
        #             for ss in fap_str.split(";"):
        #                 freq, _, resp = ss.split(",")
        #                 ff, amp, phs = [float(item) for item in resp.split(":")]
        #                 if float(freq) == self.sample_rate:
        #                     frequency = ff
        #                     amplitude = amp
        #                     phase = phs / 1e3
        #             find = True
        #         except KeyError:
        #             return fap
        # if find:
        #     freq = np.logspace(np.log10(6.00000e-04), np.log10(8.19200e03), 48)
        #     amp = np.ones(48)
        #     phases = np.zeros(48)
        #     for item_f, item_a, item_p in zip(frequency, amplitude, phase):
        #         index = np.abs(freq - item_f).argmin()
        #         freq[index] = item_f
        #         amp[index] = item_a
        #         phases[index] = item_p
        #     fap = FrequencyResponseTableFilter()
        #     fap.units_in = "milliVolt"
        #     fap.units_out = "milliVolt"
        #     fap.frequencies = freq
        #     fap.amplitudes = amp
        #     fap.phases = phases
        #     fap.name = (
        #         f"{self.header.data_logger.lower()}_{self.sample_rate:.0f}_response"
        #     )
        #     fap.comments = "data logger response read from z3d file"
        #     return fap
        # return None

    @property
    def channel_response(self):
        """
        Generate comprehensive channel response for the Z3D data.

        Creates a ChannelResponse object containing all applicable filters
        including coil response, dipole conversion, and counts-to-milliVolt
        transformation.

        Returns
        -------
        ChannelResponse
            Channel response object with appropriate filter chain for
            converting raw Z3D data to physical units.

        Notes
        -----
        The filter chain includes:
        - Coil response (for magnetic channels) or dipole filter (for electric)
        - Counts to milliVolt conversion filter
        """
        filter_list = []
        # don't have a good handle on the zen response yet.
        # if self.zen_response:
        #     filter_list.append(self.zen_response)
        frequencies = np.empty(0)
        if self.coil_response:
            filter_list.append(self.coil_response)
            frequencies = self.coil_response.frequencies
        elif self.dipole_filter:
            filter_list.append(self.dipole_filter)

        filter_list.append(self.counts2mv_filter)
        return ChannelResponse(filters_list=filter_list, frequencies=frequencies)

    @property
    def dipole_filter(self):
        """
        Create dipole conversion filter for electric field measurements.

        Generate a coefficient filter for converting electric field measurements
        from milliVolt per kilometer to milliVolt using the dipole length.

        Returns
        -------
        CoefficientFilter or None
            Filter object for dipole conversion if dipole_length is non-zero,
            None otherwise.

        Notes
        -----
        The gain is set to dipole_length/1000 to convert from mV/km to mV.
        This represents the physical dipole length scaling for electric
        field measurements.

        Examples
        --------
        >>> z3d = Z3D("/path/to/electric.Z3D")
        >>> z3d.read_all_info()
        >>> if z3d.dipole_filter is not None:
        ...     print(f"Dipole length: {z3d.dipole_length} m")
        """
        dipole = None
        # needs to be the inverse for processing
        if self.dipole_length != 0:
            dipole = CoefficientFilter()
            dipole.units_in = "milliVolt per kilometer"
            dipole.units_out = "milliVolt"
            dipole.name = f"dipole_{self.dipole_length:.2f}m"
            dipole.gain = self.dipole_length / 1000.0
            dipole.comments = "convert to electric field"
        return dipole

    def _get_gps_stamp_type(self, old_version: bool = False) -> None:
        """
        Set the correct GPS stamp data type.

        Configure GPS stamp structure for different Z3D file versions.
        Older versions use 36-bit stamps while newer versions use 64-bit stamps.

        Parameters
        ----------
        old_version : bool, default False
            If True, configure for older Z3D file format with 36-bit stamps.
            If False, use newer 64-bit stamp format.
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
    def _read_header(
        self, fn: str | Path | None = None, fid: BinaryIO | None = None
    ) -> None:
        """
        Read header information from Z3D file.

        Parameters
        ----------
        fn : str, Path, or None, optional
            Full path to Z3D file to read. If None, uses current fn attribute.
        fid : BinaryIO or None, optional
            Open file object. If provided, reads from this instead of opening fn.

        Notes
        -----
        Populates the header object's attributes and handles version-specific
        logic for older Z3D file formats.

        Examples
        --------
        Read header from file path:

        >>> z3d = Z3D()
        >>> z3d._read_header("/path/to/file.Z3D")

        Read header from open file object:

        >>> with open("/path/to/file.Z3D", 'rb') as f:
        ...     z3d._read_header(fid=f)
        """
        if fn is not None:
            self.fn = fn
        self.header.read_header(fn=self.fn, fid=fid)
        if self.header.old_version:
            if self.header.box_number is None:
                self.header.box_number = "6666"

    # ======================================
    def _read_schedule(
        self, fn: str | Path | None = None, fid: BinaryIO | None = None
    ) -> None:
        """
        Read schedule information from Z3D file.

        Parameters
        ----------
        fn : str, Path, or None, optional
            Full path to Z3D file to read. If None, uses current fn attribute.
        fid : BinaryIO or None, optional
            Open file object. If provided, reads from this instead of opening fn.

        Notes
        -----
        Populates the schedule object's attributes. For older file versions,
        extracts schedule information from the header.

        Examples
        --------
        Read schedule from file path:

        >>> z3d = Z3D()
        >>> z3d._read_schedule("/path/to/file.Z3D")

        Read schedule from open file object:

        >>> with open("/path/to/file.Z3D", 'rb') as f:
        ...     z3d._read_schedule(fid=f)
        """
        if fn is not None:
            self.fn = fn
        self.schedule.read_schedule(fn=self.fn, fid=fid)
        if self.header.old_version:
            self.schedule.initial_start = MTime(
                time_stamp=self.header.schedule, gps_time=True
            )

    # ======================================
    def _read_metadata(
        self, fn: str | Path | None = None, fid: BinaryIO | None = None
    ) -> None:
        """
        Read metadata information from Z3D file.

        Parameters
        ----------
        fn : str, Path, or None, optional
            Full path to Z3D file to read. If None, uses current fn attribute.
        fid : BinaryIO or None, optional
            Open file object. If provided, reads from this instead of opening fn.

        Notes
        -----
        Populates the metadata object's attributes. For older file versions,
        sets schedule metadata length to 0.

        Examples
        --------
        Read metadata from file path:

        >>> z3d = Z3D()
        >>> z3d._read_metadata("/path/to/file.Z3D")

        Read metadata from open file object:

        >>> with open("/path/to/file.Z3D", 'rb') as f:
        ...     z3d._read_metadata(fid=f)
        """
        if fn is not None:
            self.fn = fn
        if self.header.old_version:
            self.metadata._schedule_metadata_len = 0
        self.metadata.read_metadata(fn=self.fn, fid=fid)

    # =====================================
    def read_all_info(self) -> None:
        """
        Read header, schedule, and metadata from Z3D file.

        Convenience method to read all file information in one call.
        Opens the file once and reads all sections sequentially.

        Raises
        ------
        FileNotFoundError
            If the Z3D file does not exist.
        """
        with open(self.fn, "rb") as file_id:
            self._read_header(fid=file_id)
            self._read_schedule(fid=file_id)
            self._read_metadata(fid=file_id)

    def _find_first_gps_flag(self, fid: BinaryIO) -> int:
        """
        Find the first GPS flag in the file.

        The GPS flag should be at the end of the metadata, but sometimes
        there are extra bytes. This method searches byte by byte to find
        the first GPS flag.

        Parameters
        ----------
        fid : BinaryIO
            File object positioned after metadata.

        Returns
        -------
        int
            File position of the first GPS flag.

        Notes
        -----
        Includes failsafe to prevent infinite loops by limiting search
        to first 15000 bytes.
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

    def _read_raw_string(self, fid: BinaryIO) -> np.ndarray:
        """
        Read raw binary data from Z3D file into array.

        Reads the entire data portion of the file as 32-bit integers,
        starting from after the metadata section.

        Parameters
        ----------
        fid : BinaryIO
            Open file object positioned after metadata.

        Returns
        -------
        numpy.ndarray
            Array of int32 values containing both data and GPS stamps.
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
        logger.info(f"read {data_count} samples from file {self.fn}")

        # # now we need to trim off the extra zeros at the end
        # data = data[:data_count]
        return data

    def _unpack_data(self, data: np.ndarray, gps_stamp_index: list[int]) -> np.ndarray:
        """
        Unpack GPS stamps from raw data array and extract clean time series.

        Extract GPS timestamp information from the raw data array at
        specified indices and return a contiguous array of time series data
        with GPS stamps removed and all data blocks concatenated.

        Parameters
        ----------
        data : numpy.ndarray
            Raw data array containing both time series and GPS stamps.
        gps_stamp_index : list of int
            Indices where GPS stamps are located in the data array.

        Returns
        -------
        numpy.ndarray
            Clean time series data array with GPS stamps removed and
            all data blocks concatenated, including final block after last GPS stamp.
        """
        if len(gps_stamp_index) == 0:
            return data

        # Extract GPS stamps and calculate block lengths
        time_series_blocks = []

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
                # Extract GPS stamp
                gps_str = struct.pack(
                    "<" + "i" * int(self._gps_bytes),
                    *data[int(gps_find) : int(gps_find + self._gps_bytes)],
                )
                self.gps_stamps[ii] = np.frombuffer(gps_str, dtype=self._gps_dtype)

                # Calculate block length and extract data block before this GPS stamp
                if ii > 0:
                    block_start = gps_stamp_index[ii - 1] + int(self._gps_bytes)
                    block_end = gps_find
                    block_len = block_end - block_start
                    self.gps_stamps[ii]["block_len"] = block_len

                    # Extract the data block
                    if block_len > 0:
                        time_series_blocks.append(data[block_start:block_end])
                elif ii == 0:
                    # First GPS stamp - data before it (if any)
                    self.gps_stamps[ii]["block_len"] = 0
                    if gps_find > 0:
                        time_series_blocks.append(data[:gps_find])

        # Handle the final block of data after the last GPS stamp
        if len(gps_stamp_index) > 0:
            last_gps_pos = gps_stamp_index[-1]
            remaining_data_start = last_gps_pos + int(self._gps_bytes)

            if remaining_data_start < data.size:
                final_block = data[remaining_data_start:]
                # Only include final block if it contains significant non-zero data
                non_zero_count = np.count_nonzero(final_block)
                if non_zero_count > 0:
                    self.logger.debug(
                        f"Including final block with {final_block.size} samples "
                        f"({non_zero_count} non-zero)"
                    )
                    time_series_blocks.append(final_block)

        # Concatenate all blocks to form clean time series
        if time_series_blocks:
            clean_data = np.concatenate(time_series_blocks)
            # Remove any remaining zeros
            clean_data = clean_data[np.nonzero(clean_data)]
        else:
            clean_data = np.array([], dtype=data.dtype)

        return clean_data

    # ======================================
    def read_z3d(self, z3d_fn: str | Path | None = None) -> None:
        """
        Read and parse Z3D file data.

        Comprehensive method to read a Z3D file and populate all object attributes.
        Performs the following operations:

        1. Read file as chunks of 32-bit integers
        2. Extract and validate GPS stamps
        3. Check GPS time stamp consistency (1 second intervals)
        4. Verify data block lengths match sampling rate
        5. Convert GPS time to seconds relative to GPS week
        6. Skip initial buffered data (first 2 seconds)
        7. Populate time series array with non-zero data

        Parameters
        ----------
        z3d_fn : str, Path, or None, optional
            Path to Z3D file to read. If None, uses current fn attribute.

        Raises
        ------
        ZenGPSError
            If data is too short or GPS timing issues prevent parsing.

        Examples
        --------
        >>> z3d = Z3D(r"/path/to/data/station_20150522_080000_256_EX.Z3D")
        >>> z3d.read_z3d()
        >>> print(f"Read {z3d.time_series.size} data points")
        """
        if z3d_fn is not None:
            self.fn = z3d_fn
        self.logger.debug(f"Reading {self.fn}")
        st = MTime().now()

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
        self.time_series = self._unpack_data(data, gps_stamp_find)

        # validate everything
        if not self.validate_gps_time():
            self.logger.debug(
                f"GPS stamps are not 1 second apart for file {self.fn.name}."
            )
        if not self.validate_time_blocks():
            self.logger.debug(
                f"Time block between stamps was not the sample rate for file {self.fn.name}"
            )
        self.convert_gps_time()
        self.zen_schedule = self.check_start_time()
        self.logger.debug(f"found {self.gps_stamps.shape[0]} GPS time stamps")
        self.logger.debug(f"found {self.time_series.size} data points")

        # time it
        et = MTime().now()
        read_time = et - st
        self.logger.info(f"Reading data took: {read_time:.3f} seconds")

    # =================================================
    def get_gps_stamp_index(
        self, ts_data: np.ndarray, old_version: bool = False
    ) -> list[int]:
        """
        Locate GPS time stamp indices in time series data.

        Searches for GPS flag patterns in the data array. For newer files,
        verifies that flag_1 follows flag_0.

        Parameters
        ----------
        ts_data : numpy.ndarray
            Time series data array containing GPS stamps.
        old_version : bool, default False
            If True, only searches for single GPS flag (old format).
            If False, validates flag pairs (new format).

        Returns
        -------
        list of int
            List of indices where GPS stamps are located.
        """
        # find the gps stamps
        gps_stamp_find = np.where(ts_data == self._gps_flag_0)[0]

        if old_version is False:
            gps_stamp_find = [
                gps_find
                for gps_find in gps_stamp_find
                if ts_data[gps_find + 1] == self._gps_flag_1
            ]
        return list(gps_stamp_find)

    # =================================================
    def trim_data(self) -> None:
        """
        Trim the first 2 seconds of data due to SD buffer issues.

        Remove the first 2 GPS stamps and corresponding time series data
        to account for SD card buffering artifacts in early data.

        Notes
        -----
        This method may be deprecated after field testing confirms
        the buffer behavior is consistent across all instruments.

        .. deprecated::
           This method will be deprecated after field testing.
        """
        # the block length is the number of data points before the time stamp
        # therefore the first block length is 0.  The indexing in python
        # goes to the last index - 1 so we need to put in 3
        ts_skip = self.gps_stamps["block_len"][0:3].sum()
        self.gps_stamps = self.gps_stamps[2:]
        self.gps_stamps[0]["block_len"] = 0
        self.time_series = self.time_series[ts_skip:]

    # =================================================
    def check_start_time(self) -> MTime:
        """
        Validate scheduled start time against first GPS stamp.

        Compare the scheduled start time from the file header with
        the actual first GPS timestamp to identify timing discrepancies.

        Returns
        -------
        MTime
            UTC start time from the first valid GPS stamp.

        Notes
        -----
        Logs warnings if the difference exceeds the maximum allowed
        time difference (default 20 seconds).
        """
        # make sure the time is in gps time
        zen_start_utc = self.get_UTC_date_time(
            self.header.gpsweek, float(self.gps_stamps["time"][0])
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
    def validate_gps_time(self) -> bool:
        """
        Validate that GPS time stamps are consistently 1 second apart.

        Returns
        -------
        bool
            True if all GPS stamps are properly spaced, False otherwise.

        Notes
        -----
        Logs debug information for any stamps that are more than 1 second apart.
        """
        # need to put the gps time into seconds
        t_diff = np.diff(self.gps_stamps["time"]) / 1024

        bad_times = np.where(abs(t_diff) > 1)[0]
        if len(bad_times) > 0:
            for bb in bad_times:
                self.logger.warning(
                    f"Data block {bb} has a time difference between GPS stamps > 1 second [{t_diff[bb]} s]"
                )
            return False
        return True

    # ===================================================
    def validate_time_blocks(self) -> bool:
        """
        Validate GPS time stamps and verify data block lengths.

        Check that each GPS stamp block contains the expected number
        of data points (should equal sample rate for 1-second blocks).

        Returns
        -------
        bool
            True if all blocks have correct length, False otherwise.

        Notes
        -----
        If bad blocks are detected near the beginning (index < 5),
        this method will automatically skip those blocks and trim
        the time series data accordingly.
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
                self.logger.warning(f"Skipped first {ts_skip} points in time series")
                return True
            for bb in bad_blocks:
                self.logger.warning(
                    f"Data block {bb} has {self.gps_stamps['block_len'][bb+1]} samples, "
                    f"expected {self.header.ad_rate} samples."
                )
            return False
        return True

    # ==================================================
    def convert_gps_time(self) -> None:
        """
        Convert GPS time integers to floating point seconds.

        Transform GPS time from integer format to float and convert
        from GPS time units to seconds relative to the GPS week.

        Notes
        -----
        GPS time is initially stored as integers in units of 1/1024 seconds.
        This method converts to floating point seconds and applies the
        necessary scaling factors.
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
    def convert_counts_to_mv(self, data: np.ndarray) -> np.ndarray:
        """
        Convert time series data from counts to milliVolt.

        Parameters
        ----------
        data : numpy.ndarray
            Time series data in digital counts.

        Returns
        -------
        numpy.ndarray
            Time series data converted to milliVolt.
        """
        data *= self._counts_to_mv_conversion
        return data

    def convert_mv_to_counts(self, data: np.ndarray) -> np.ndarray:
        """
        Convert time series data from milliVolt to counts.

        Parameters
        ----------
        data : numpy.ndarray
            Time series data in milliVolt.

        Returns
        -------
        numpy.ndarray
            Time series data converted to digital counts.

        Notes
        -----
        Assumes no other scaling has been applied to the data.
        """
        data /= self._counts_to_mv_conversion
        return data

    # ==================================================
    def get_gps_time(self, gps_int: int, gps_week: int = 0) -> tuple[float, int]:
        """
        Convert GPS integer timestamp to seconds and GPS week.

        Parameters
        ----------
        gps_int : int
            Integer from the GPS time stamp line.
        gps_week : int, default 0
            Relative GPS week. If seconds exceed one week, this is incremented.

        Returns
        -------
        tuple[float, int]
            GPS time in seconds from beginning of GPS week, and updated GPS week.

        Notes
        -----
        GPS integers are in units of 1/1024 seconds. This method handles
        week rollovers when seconds exceed 604800.
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
    def get_UTC_date_time(self, gps_week: int, gps_time: float) -> MTime:
        """
        Convert GPS week and time to UTC datetime.

        Calculate the actual UTC date and time of measurement from
        GPS week number and seconds within that week.

        Parameters
        ----------
        gps_week : int
            GPS week number when data was collected.
        gps_time : float
            Number of seconds from beginning of GPS week.

        Returns
        -------
        MTime
            UTC datetime object for the measurement time.

        Notes
        -----
        Automatically handles GPS time rollover when seconds exceed
        one week (604800 seconds).
        """
        # need to check to see if the time in seconds is more than a gps week
        # if it is add 1 to the gps week and reduce the gps time by a week
        if gps_time > self._week_len:
            gps_week += 1
            gps_time -= self._week_len
        # compute seconds using weeks and gps time
        # Convert gps_time to Python float to avoid precision issues with numpy types
        utc_seconds = (
            self._gps_epoch.epoch_seconds
            + (gps_week * self._week_len)
            + float(gps_time)
        )

        # compute date and time from seconds and return a datetime object
        # easier to manipulate later, must be in nanoseconds
        return MTime(time_stamp=utc_seconds, gps_time=True)

    # =================================================
    def to_channelts(self) -> ChannelTS:
        """
        Convert Z3D data to ChannelTS time series object.

        Create a ChannelTS object populated with the time series data
        and all associated metadata from the Z3D file.

        Returns
        -------
        ChannelTS
            Time series object with data, metadata, and instrument response.
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
    """Exception raised for GPS timing errors in Z3D files."""


class ZenSamplingRateError(Exception):
    """Exception raised for sampling rate inconsistencies."""


class ZenInputFileError(Exception):
    """Exception raised for Z3D file input/reading errors."""


def read_z3d(
    fn: str | Path,
    calibration_fn: str | Path | None = None,
    logger_file_handler: Any | None = None,
) -> ChannelTS | None:
    """
    Read a Z3D file and return a ChannelTS object.

    Convenience function to read Z3D files with error handling.

    Parameters
    ----------
    fn : str or Path
        Path to the Z3D file to read.
    calibration_fn : str, Path, or None, optional
        Path to calibration file. Default is None.
    logger_file_handler : optional
        Logger file handler to add to Z3D logger. Default is None.

    Returns
    -------
    ChannelTS or None
        Time series object if successful, None if GPS timing errors occur.

    Examples
    --------
    >>> ts = read_z3d("/path/to/data/station_EX.Z3D")
    >>> if ts is not None:
    ...     print(f"Read {ts.n_samples} samples")
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
