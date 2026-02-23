# -*- coding: utf-8 -*-
"""
LEMI-423 Reader
===============

Read LEMI-423 broadband magnetotelluric binary files (*.B423).

Stores RAW counts with calibrations as unapplied filters (MTH5 standard).

:author: ben kay
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from loguru import logger

from mth5.timeseries import ChannelTS, RunTS
from mt_metadata.timeseries import Station, Run, Electric, Magnetic
from mt_metadata.timeseries.filters import (
    FrequencyResponseTableFilter,
    ChannelResponse,
    CoefficientFilter,
)
from mt_metadata.utils.mttime import MTime

# File extensions (central dispatcher lower-cases extension)
B423_EXTS = {"b423"}

class Read_Lemi_Header:
    """
    Parse LEMI-423 ASCII header (1024 bytes).

    Header Format
    -------------
    The first 1024 bytes of each *.B423 file contain ASCII metadata:

    **Lines 0-2: Instrument Identification**
        - Line 0: ``%LEMI423 #0036`` (instrument serial number)
        - Line 1: ``%FIRMWARE Ver.2.1`` (firmware version)
        - Line 2: ``%MADE in UKRAINE`` (manufacturer)

    **Line 3: (blank)**

    **Lines 4-8: Deployment Time and Power Status**
        - Line 4: ``%Date 2021/06/24`` (deployment date, YYYY/MM/DD)
        - Line 5: ``%Time 03:53:42`` (deployment time, HH:MM:SS UTC)
        - Line 6: ``%Ubat 13.16V`` (battery voltage)
        - Line 7: ``%Current 101.7mA`` (current draw)
        - Line 8: ``%Free 30424MB`` (free storage space)

    **Lines 9-11: Geographic Location**
        - Line 9: ``%Lat 2944.90064,S`` (latitude in DDMM.MMMMM format, N/S)
        - Line 10: ``%Lon 13900.11658,E`` (longitude in DDDMM.MMMMM format, E/W)
        - Line 11: ``%Alt 119.9,m 12 2`` (altitude in meters)

    **Line 12: (blank)**

    **Lines 13+: Linear Calibration Coefficients**
        Magnetic channels (Bx, By, Bz):
            - ``%Kmx = 2.909985e-06`` (gain for Bx: counts → nT)
            - ``%Kmy = 2.909481e-06`` (gain for By: counts → nT)
            - ``%Kmz = 2.908610e-06`` (gain for Bz: counts → nT)
            - ``%Ax = -5.002100e+01`` (offset for Bx in nT)
            - ``%Ay = -4.990500e+01`` (offset for By in nT)
            - ``%Az = -4.994700e+01`` (offset for Bz in nT)

        Electric channels (Ex, Ey):
            - ``%Ke1 = 2.910737e-04`` (gain for Ex: counts → V)
            - ``%Ke2 = 2.909547e-04`` (gain for Ey: counts → V)
            - ``%Ae1 = -5.004800e+03`` (offset for Ex in V)
            - ``%Ae2 = -4.958000e+03`` (offset for Ey in V)

        Auxiliary channels (for direct-to-PC recording; not in SD card *.B423 files):
            - ``%Kte, %Ate`` (temperature calibration)
            - ``%KUin, %AUin`` (input voltage calibration)
            - ``%Kc, %Ac`` (current calibration)

    Calibration Formula
    -------------------
    All channels use linear calibration:

        **physical_value = (raw_counts × K) + A**

    Where K is the gain coefficient and A is the offset.

    Extracted Data
    --------------
    - **instrument_number** (int): Serial number from line 0
    - **firmware_version** (str): Version string from line 1
    - **deployment_time** (Timestamp): Combined date/time from lines 4-5 (UTC)
    - **latitude, longitude** (float): Decimal degrees (converted from DDMM.MMMMM)
    - **elevation** (float): Meters above sea level from line 11
    - **coefficients** (dict): All K and A coefficients (Kmx, Kmy, Kmz, Ke1, Ke2, Ax, Ay, Az, Ae1, Ae2, etc.)
    """
    def __init__(self, binary_file: Union[str, Path]):
        self.binary_file = str(binary_file)
        self.instrument_number = None
        self.firmware_version = None
        self.deployment_time = None
        self.latitude = self.longitude = self.elevation = None
        self.coefficients: Dict[str, float] = {}

    def _read_header(self) -> List[str]:
        with open(self.binary_file, 'rb') as f:
            return f.read(1024).decode(errors='ignore').splitlines()

    def _extract_instrument_number(self, header):
        self.instrument_number = int(header[0].split('#')[-1]) if '#' in header[0] else None

    def _extract_firmware_version(self, header):
        # Line 1: "%FIRMWARE Ver.2.1" -> extract "2.1"
        if len(header) > 1 and 'FIRMWARE' in header[1]:
            parts = header[1].split()
            if len(parts) >= 2:
                # Extract version after "Ver."
                self.firmware_version = parts[-1].replace('Ver.', '')

    def _extract_deployment_time(self, header):
        date, time = header[4].split()[-1], header[5].split()[-1]
        # e.g., "YYYY/MM/DD HH:MM:SS"
        self.deployment_time = pd.to_datetime(f"{date} {time}", format="%Y/%m/%d %H:%M:%S", utc=True)

    def _extract_coefficients(self, header):
        self.coefficients = {
            line.split("=")[0].strip().lstrip('%'): float(line.split("=")[1])
            for line in header[13:] if "=" in line
        }

    def _extract_coordinates(self, header):
        lat, lat_dir = header[9].split()[-1].split(',')
        lon, lon_dir = header[10].split()[-1].split(',')
        self.latitude = (int(lat[:2]) + float(lat[2:]) / 60) * (-1 if lat_dir.strip() == 'S' else 1)
        self.longitude = (int(lon[:3]) + float(lon[3:]) / 60) * (-1 if lon_dir.strip() == 'W' else 1)
        self.elevation = float(header[11].split(',')[0].split()[-1])

    def read(self) -> Dict:
        header = self._read_header()
        self._extract_instrument_number(header)
        self._extract_firmware_version(header)
        self._extract_deployment_time(header)
        self._extract_coefficients(header)
        self._extract_coordinates(header)
        return {
            "instrument_number": self.instrument_number,
            "firmware_version": self.firmware_version,
            "deployment_time": self.deployment_time,  # pandas.Timestamp(tz=UTC)
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "coefficients": self.coefficients,
        }


class Read_Lemi_Data:
    """
    Read LEMI-423 binary file and return RAW counts (no calibration applied).

    Binary Format
    -------------
    After the 1024-byte ASCII header, binary data is structured as:

    **30 bytes per sample** (little-endian):

    ======  =======  ======  =====================================
    Offset  Bytes    Type    Field
    ======  =======  ======  =====================================
    0       4        uint32  time (Unix timestamp, seconds)
    4       2        uint16  tick (sub-second counter, millisecond resolution)
    6       4        int32   Bx (raw counts)
    10      4        int32   By (raw counts)
    14      4        int32   Bz (raw counts)
    18      4        int32   Ex (raw counts)
    22      4        int32   Ey (raw counts)
    26      1        int8    gps (deviation from PPS)
    27      1        uint8   gps (PLL accuracy)
    28      2        int16   CRC (checksum)
    ======  =======  ======  =====================================

    Data Extraction
    ---------------
    - **Timestamp**: Constructed from time + tick (millisecond resolution)
    - **Magnetic/Electric**: RAW counts returned as-is (int32)
    - **Unused fields**: sync, stage, CRC are read but discarded

    Returns
    -------
    pandas.DataFrame
        Time-indexed DataFrame with columns Bx, By, Bz, Ex, Ey (RAW counts, int32).
    """
    
    def __init__(self, binary_file: Union[str, Path], coefficients: Dict[str, float]):
        self.binary_file = str(binary_file)
        self.coefficients = coefficients

    def read_dataframe(self) -> pd.DataFrame:
        # Binary: 30 bytes/sample, little-endian, 1024-byte header
        binary_format = np.dtype([
            ('time', '<u4'), ('tick', '<u2'),
            ('Bx', '<i4'), ('By', '<i4'), ('Bz', '<i4'),
            ('Ex', '<i4'), ('Ey', '<i4'),
            ('sync', '<i1'), ('stage', '<u1'), ('CRC', '<i2'),
        ])
        with open(self.binary_file, 'rb') as f:
            f.read(1024)  # skip header
            arr = np.fromfile(f, dtype=binary_format)
        if arr.size == 0:
            return pd.DataFrame(columns=["Bx","By","Bz","Ex","Ey"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="time")
            )

        df = pd.DataFrame(arr)

        # Determine sample rate from tick counter range
        # Tick resets to 0 when timestamp increments, so max(tick) + 1 = sample rate
        tick_max = df['tick'].max()
        self.sample_rate = float(tick_max + 1) if tick_max > 0 else None

        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True) + pd.to_timedelta(df['tick'], unit='ms')
        df.set_index('time', inplace=True)

        # Return RAW counts (no calibration applied)
        return df[['Bx','By','Bz','Ex','Ey']].sort_index()


def read_lemi_coil_response(calibration_fn, coil_number=None):
    """
    Read LEMI-120 coil calibration from .rsp file.

    :param calibration_fn: Path to .rsp calibration file
    :type calibration_fn: str or Path
    :param coil_number: Optional coil serial number
    :type coil_number: str, optional
    :return: Frequency response table filter
    :rtype: FrequencyResponseTableFilter

    **File Format**: ASCII with 2 header lines, then columns: freq (Hz), amp, phase (deg)
    """
    calibration_fn = Path(calibration_fn)

    # Read the file (skip first 2 header lines)
    cal_data = np.loadtxt(calibration_fn, skiprows=2)

    # Create frequency response filter
    fap = FrequencyResponseTableFilter()
    fap.frequencies = cal_data[:, 0]  # Hz
    fap.amplitudes = cal_data[:, 1]   # Normalized
    fap.phases = np.deg2rad(cal_data[:, 2])  # Convert degrees to radians
    fap.units_in = "nanotesla"
    fap.units_out = "millivolts"
    fap.name = f"lemi_120_{coil_number}_response" if coil_number else "lemi_120_response"
    fap.type = "frequency response table"
    fap.calibration_date = "1970-01-01T00:00:00+00:00"  # Default
    fap.comments = f"LEMI-120 coil response from {calibration_fn.name}"

    return fap


def create_lemi423_linear_calibration_filter(
    component: str, k_coeff: float, a_coeff: float
) -> CoefficientFilter:
    """
    Create linear calibration filter for LEMI-423 channels.

    Formula: physical_value = (raw_counts × K) + A

    :param component: Channel component (hx, hy, hz, ex, ey)
    :param k_coeff: Gain coefficient from header
    :param a_coeff: Offset coefficient from header
    :return: Coefficient filter (counts → nT or V)
    """
    coeff_filter = CoefficientFilter()
    coeff_filter.name = f"lemi423_linear_{component}"
    coeff_filter.units_in = "counts"

    # Set output units based on channel type
    if component in ["hx", "hy", "hz"]:
        coeff_filter.units_out = "nanotesla"
    else:  # ex, ey
        coeff_filter.units_out = "volts"

    coeff_filter.gain = k_coeff
    coeff_filter.offset = a_coeff
    coeff_filter.comments = (
        f"LEMI-423 linear calibration: {component.upper()} = "
        f"(counts × {k_coeff}) + {a_coeff}"
    )

    return coeff_filter


# ---------- MTH5-facing reader ----------
class LEMI423Reader:
    """
    Read LEMI-423 binary files (*.B423) following MTH5 standard.

    Stores RAW counts with calibrations as unapplied filters.

    :param files: Single file path or list of *.B423 files
    :type files: str, Path, or list
    :param kwargs: Additional options
    :type kwargs: dict

    :Keyword Arguments:
        * **calibration_fn** (str) - Path to LEMI-120 .rsp file (optional)
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 0)
        * **station_id** (str) - Station identifier (optional)

    **Filter Chain**:
        - All channels: linear calibration (counts → nT or V)
        - Magnetic (optional): LEMI-120 coil response (nT → mV)
    """

    def __init__(self, files: List[Union[str, Path]], **kwargs):
        self.logger = logger
        self.files = [Path(f) for f in (files if isinstance(files, list) else [files])]
        self.sample_rate = kwargs.get('sample_rate', None)
        self.dipole_length_ex = kwargs.get('dipole_length_ex', 0)
        self.dipole_length_ey = kwargs.get('dipole_length_ey', 0)
        self.station_id = kwargs.get('station_id', None)
        self.calibration_fn = kwargs.get('calibration_fn', None)
        self.data = None
        self.header = None

    def __str__(self):
        lines = ["LEMI-423 Reader", "-" * 20]
        if self.header:
            lines.append(f"Instrument: LEMI-423 #{self.header.get('instrument_number', 'unknown')}")
            firmware = self.header.get('firmware_version', '2.1')
            lines.append(f"Firmware:   Ver.{firmware}")
        if self.data is not None:
            lines.append(f"Start:      {self.start.isoformat()}")
            lines.append(f"End:        {self.end.isoformat()}")
            lines.append(f"N samples:  {self.n_samples}")
            lines.append(f"Sample rate: {self.sample_rate} Hz")
            lines.append(f"Latitude:   {self.latitude} (degrees)")
            lines.append(f"Longitude:  {self.longitude} (degrees)")
            lines.append(f"Elevation:  {self.elevation} m")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def _has_data(self):
        """Check if data has been read"""
        return self.data is not None

    @property
    def start(self):
        """Start time of data collection"""
        if self._has_data():
            return MTime(self.data.index[0])
        elif self.header:
            return MTime(self.header.get('deployment_time'))
        return None

    @property
    def end(self):
        """End time of data collection"""
        if self._has_data():
            return MTime(self.data.index[-1])
        return None

    @property
    def n_samples(self):
        """Number of samples in the data"""
        if self._has_data():
            return self.data.shape[0]
        return 0

    @property
    def latitude(self):
        """Latitude from header in decimal degrees (WGS84)"""
        if self.header:
            return self.header.get('latitude')
        return None

    @property
    def longitude(self):
        """Longitude from header in decimal degrees (WGS84)"""
        if self.header:
            return self.header.get('longitude')
        return None

    @property
    def elevation(self):
        """Elevation from header in meters"""
        if self.header:
            return self.header.get('elevation')
        return None

    def _build_station_metadata(self) -> Station:
        """
        Build station metadata without mutation side effects.

        Internal helper method for constructing Station object.
        Does not call other properties to avoid circular dependencies.

        :return: Station metadata object
        :rtype: :class:`mt_metadata.timeseries.Station`
        """
        s = Station()
        if self.header:
            s.location.latitude = self.latitude
            s.location.longitude = self.longitude
            s.location.elevation = self.elevation
            # Determine station ID from multiple sources (priority order):
            if self.station_id:
                # 1. Explicitly provided via kwargs
                s.id = self.station_id
            elif self.files:
                # 2. Use parent folder name (typical data organization pattern)
                s.id = self.files[0].parent.name
            else:
                # 3. Fall back to instrument number
                instrument_num = self.header.get('instrument_number', '')
                s.id = f"LEMI{instrument_num:04d}" if instrument_num else ""
            # LEMI instruments typically use geomagnetic reference frame
            s.orientation.reference_frame = "geomagnetic"
        if self._has_data():
            s.time_period.start = self.start.isoformat()
            s.time_period.end = self.end.isoformat()
        return s

    def _build_run_metadata(self) -> Run:
        """
        Build run metadata without mutation side effects.

        Internal helper method for constructing Run object.
        Does not call other properties to avoid circular dependencies.

        :return: Run metadata object
        :rtype: :class:`mt_metadata.timeseries.Run`
        """
        r = Run()
        r.id = "a"
        r.data_logger.model = "LEMI-423"
        r.data_logger.manufacturer = "LEMI"
        r.data_logger.type = "broadband"
        r.data_type = "MTBB"  # Magnetotelluric Broadband

        if self.header:
            instrument_num = self.header.get('instrument_number', '')
            r.data_logger.id = str(instrument_num) if instrument_num else ""

            # Use firmware version from file header, default to "2.1" if not found
            firmware_version = self.header.get('firmware_version', '2.1')
            r.data_logger.firmware.version = firmware_version

        if self._has_data():
            r.sample_rate = self.sample_rate
            r.time_period.start = self.start.isoformat()
            r.time_period.end = self.end.isoformat()

            # Add channel metadata for each component
            for ch_h in ["hx", "hy", "hz"]:
                r.add_channel(Magnetic(component=ch_h))
            for ch_e in ["ex", "ey"]:
                r.add_channel(Electric(component=ch_e))

        return r

    @property
    def station_metadata(self):
        """
        Station metadata as :class:`mt_metadata.timeseries.Station`

        Returns station-level information including location and time period.
        Public API property for external access.
        """
        return self._build_station_metadata()

    @property
    def run_metadata(self):
        """
        Run metadata as :class:`mt_metadata.timeseries.Run`

        Returns run-level information including data logger details and channels.
        Public API property for external access.
        """
        return self._build_run_metadata()

    def _get_channel_metadata(self, component: str, channel_number: int):
        """
        Get metadata for a specific channel

        :param component: channel component name (hx, hy, hz, ex, ey)
        :type component: str
        :param channel_number: channel number (1-5)
        :type channel_number: int
        :return: channel metadata object
        :rtype: Magnetic or Electric
        """
        if component in ["hx", "hy", "hz"]:
            ch_metadata = Magnetic()
            ch_metadata.type = "magnetic"
            # Store RAW counts (following MTH5 standard like Phoenix, Zen, NIMS, Metronix)
            ch_metadata.units = "counts"
            # Set default azimuths (can be updated later if known)
            azimuth_map = {"hx": 0, "hy": 90, "hz": 0}
            tilt_map = {"hx": 0, "hy": 0, "hz": 90}
            ch_metadata.measurement_azimuth = azimuth_map.get(component, 0)
            ch_metadata.measurement_tilt = tilt_map.get(component, 0)
            ch_metadata.sensor.manufacturer = "LEMI"
            ch_metadata.sensor.model = "LEMI-120"
            ch_metadata.sensor.type = "induction coil"
            # Use instrument number as sensor ID if available
            if self.header:
                instrument_num = self.header.get('instrument_number', '')
                ch_metadata.sensor.id = str(instrument_num) if instrument_num else ""
        else:
            ch_metadata = Electric()
            ch_metadata.type = "electric"
            # Store RAW counts (following MTH5 standard)
            ch_metadata.units = "counts"
            # Set default azimuths (can be updated later if known)
            azimuth_map = {"ex": 0, "ey": 90}
            ch_metadata.measurement_azimuth = azimuth_map.get(component, 0)
            ch_metadata.measurement_tilt = 0
            # Use dipole length from kwargs if provided, otherwise 0
            if component == "ex":
                ch_metadata.dipole_length = self.dipole_length_ex
            elif component == "ey":
                ch_metadata.dipole_length = self.dipole_length_ey
            # Note: positive.id, negative.id should be set by user after reading

        ch_metadata.component = component
        ch_metadata.channel_number = channel_number

        if self._has_data():
            ch_metadata.sample_rate = self.sample_rate
            ch_metadata.time_period.start = self.start.isoformat()
            ch_metadata.time_period.end = self.end.isoformat()

        return ch_metadata

    def _read_one(self, path: Path) -> Tuple[pd.DataFrame, Dict]:
        """Read a single B423 file"""
        hdr = Read_Lemi_Header(path).read()
        data_reader = Read_Lemi_Data(path, hdr["coefficients"])
        df = data_reader.read_dataframe()
        # Add sample_rate to header if detected from tick counter
        if hasattr(data_reader, 'sample_rate') and data_reader.sample_rate:
            hdr['sample_rate'] = data_reader.sample_rate
        return df, hdr

    def read(self) -> RunTS:
        """
        Read LEMI-423 file(s) and return a RunTS with channels hx, hy, hz, ex, ey.

        Concatenates multiple files by timestamp, removes duplicates, and preserves gaps.

        :return: RunTS object containing ChannelTS for each component
        :rtype: :class:`mth5.timeseries.RunTS`

        :Example:
            >>> reader = LEMI423Reader(['file1.B423', 'file2.B423'])
            >>> run_ts = reader.read()
        """
        # Read all files, gather per-file metadata (use first header for station/run attrs)
        parts: List[pd.DataFrame] = []
        headers: List[Dict] = []
        for p in self.files:
            df, h = self._read_one(p)
            parts.append(df)
            headers.append(h)

        if not parts:
            self.logger.warning("No data found in files")
            return RunTS([])

        # Store first header as the primary metadata source
        self.header = headers[0]

        # Concatenate on time index
        full = pd.concat(parts).sort_index()
        # Drop exact duplicates
        full = full[~full.index.duplicated(keep='first')]

        # Store the full dataframe
        self.data = full

        # Use sample rate calculated from tick counter (max_tick + 1) if available
        # Otherwise fall back to median dt calculation
        if self.header.get('sample_rate'):
            self.sample_rate = self.header['sample_rate']
        elif len(full.index) > 1:
            dt = full.index.to_series().diff().dropna().dt.total_seconds().median()
            self.sample_rate = (1.0 / dt) if dt and dt > 0 else None
        else:
            self.sample_rate = None

        self.logger.info(f"Read {len(parts)} file(s), {self.n_samples} samples at {self.sample_rate} Hz")

        # Build metadata objects ONCE and reuse them
        # This avoids repeated property calls and object reconstruction
        station_meta = self._build_station_metadata()
        run_meta = self._build_run_metadata()

        # Build ChannelTS objects with proper metadata
        ch_objs: List[ChannelTS] = []
        # Map file column names to MTH5 standard channel codes
        mapping = {
            "Bx": ("hx", "magnetic", 1),
            "By": ("hy", "magnetic", 2),
            "Bz": ("hz", "magnetic", 3),
            "Ex": ("ex", "electric", 4),
            "Ey": ("ey", "electric", 5),
        }

        for src, (code, ch_type, ch_num) in mapping.items():
            if src not in full.columns:
                self.logger.warning(f"Channel {src} not found in data")
                continue

            series = full[src].to_numpy()
            ch_metadata = self._get_channel_metadata(code, ch_num)

            # **NEW**: Create calibration filter chain (following MTH5 standard)
            # All channels get linear calibration filter (counts → physical units)
            # Magnetic channels optionally get LEMI-120 coil response if calibration_fn provided
            filters_list = []

            # Get calibration coefficients from header
            if self.header:
                coeffs = self.header.get('coefficients', {})

                # Map channel code to coefficient keys
                coeff_map = {
                    'hx': ('Kmx', 'Ax'),
                    'hy': ('Kmy', 'Ay'),
                    'hz': ('Kmz', 'Az'),
                    'ex': ('Ke1', 'Ae1'),
                    'ey': ('Ke2', 'Ae2'),
                }

                if code in coeff_map:
                    k_key, a_key = coeff_map[code]
                    k_val = coeffs.get(k_key, 1.0)
                    a_val = coeffs.get(a_key, 0.0)

                    # Create linear calibration filter (counts → nT or V)
                    linear_filter = create_lemi423_linear_calibration_filter(
                        code, k_val, a_val
                    )
                    filters_list.append(linear_filter)

            # Add LEMI-120 coil response for magnetic channels (if provided)
            if code in ["hx", "hy", "hz"] and self.calibration_fn is not None:
                # Get instrument number for coil ID
                coil_id = None
                if self.header:
                    instrument_num = self.header.get('instrument_number', '')
                    coil_id = str(instrument_num) if instrument_num else None

                # Read LEMI-120 coil response filter from .rsp file
                coil_filter = read_lemi_coil_response(
                    self.calibration_fn,
                    coil_number=coil_id
                )
                filters_list.append(coil_filter)

            # Create ChannelResponse with all filters
            channel_response = None
            if filters_list:
                channel_response = ChannelResponse(filters_list=filters_list)

                # Update metadata to reference all filters
                ch_metadata.filter.name = [f.name for f in filters_list]
                ch_metadata.filter.applied = [False] * len(filters_list)

            # Create ChannelTS object with channel_response
            ch = ChannelTS(
                channel_type=ch_type,
                data=series,
                channel_metadata=ch_metadata,
                run_metadata=run_meta,          # Reuse cached object
                station_metadata=station_meta,  # Reuse cached object
                channel_response=channel_response,  # Pass ChannelResponse object
            )

            ch_objs.append(ch)

        return RunTS(
            array_list=ch_objs,
            station_metadata=station_meta,  # Reuse cached object
            run_metadata=run_meta,          # Reuse cached object
        )


def read_lemi423(fn: Union[str, Path, List[Union[str, Path]]], **kwargs) -> RunTS:
    """
    Read LEMI-423 binary files (*.B423) and return RunTS.

    Stores RAW counts with calibrations as unapplied filters (MTH5 standard).

    :param fn: Single file path or list of *.B423 files
    :type fn: str, Path, or list
    :param kwargs: Additional options
    :type kwargs: dict

    :Keyword Arguments:
        * **calibration_fn** (str) - Path to LEMI-120 .rsp file (optional)
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 0)
        * **station_id** (str) - Station identifier (optional)

    :return: RunTS with channels hx, hy, hz, ex, ey (RAW counts)
    :rtype: RunTS

    :Example:
        >>> from mth5.io.lemi import read_lemi423
        >>>
        >>> # Read without coil calibration (only linear calibration)
        >>> run_ts = read_lemi423('/path/to/MT001/data.B423')
        >>>
        >>> # Read with LEMI-120 coil calibration
        >>> run_ts = read_lemi423(
        ...     '/path/to/MT001/data.B423',
        ...     calibration_fn='/path/to/l120n.rsp'
        ... )
        >>>
        >>> # Read multiple files with dipole lengths and calibration
        >>> run_ts = read_lemi423(
        ...     ['file1.B423', 'file2.B423'],
        ...     calibration_fn='l120n.rsp',
        ...     dipole_length_ex=50.0,
        ...     dipole_length_ey=50.0
        ... )
        >>>
        >>> # Override station ID
        >>> run_ts = read_lemi423('data.B423', station_id='CustomStation')
        >>>
        >>> # Access channel data and metadata
        >>> hx_data = run_ts.hx.to_numpy()
        >>> print(run_ts.station_metadata.id)
        >>> print(run_ts.ex.channel_metadata.dipole_length)
        >>> # Check if coil calibration was included
        >>> if run_ts.hx.channel_metadata.filter.name:
        ...     print(f"Coil filter: {run_ts.hx.channel_metadata.filter.name[0]}")

    .. note::
        - LEMI-423 is a broadband MT instrument (data_type: "MTBB")
        - Files are concatenated by timestamp with duplicates removed
        - Gaps in data are preserved (not interpolated)
        - Linear calibration coefficients from file header are always applied
        - Coil response calibration requires explicit calibration_fn parameter
        - Electric field units are Volts (convert to V/m using dipole length)
        - Sample rate is auto-detected from time index (median dt)
        - Station ID priority: 1) kwargs, 2) parent folder name, 3) instrument number

    .. seealso::
        :func:`read_lemi_coil_response` - Read LEMI-120 .rsp calibration files
    """
    files = [fn] if isinstance(fn, (str, Path)) else list(fn)
    rdr = LEMI423Reader(files, **kwargs)
    return rdr.read()