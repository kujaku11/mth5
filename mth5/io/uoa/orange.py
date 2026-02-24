"""
================================================================================
Orange Box Magnetotelluric Reader for MTH5
================================================================================

Reads binary magnetotelluric data from legacy University of Adelaide / Flinders
University "Orange Box" long-period MT systems.

**IMPORTANT**: This reader is for the specific UoA/Flinders Orange Box configuration.

The Orange Box is a custom-built 8-channel long-period MT data logger that
records data in binary format with minimal header information.

**File Format**:
    - Binary with ASCII header (3 lines)
    - Header line 1: Sample rate (hex string, e.g., "00008CA0")
    - Header line 2: Start date/time (ASCII, e.g., "Tue Jun 16 02:01:04 2009")
    - Header line 3: Filter point (4-byte hex string, e.g., " 7A1")
    - Binary data: 8 channels per sample, variable byte lengths

**Channel Layout** (per sample, 18 bytes total):
    - Channels 0-2 (Bx, Bz, By): 3 bytes each (24-bit unsigned, big-endian)
    - Channels 3-4: 2 bytes each (16-bit unsigned, big-endian)
    - Channel 5: 1 byte (8-bit unsigned)
    - Channels 6-7 (Ey, Ex): 3 bytes each (24-bit unsigned, big-endian)
    - 1 extra byte (padding/sync)

**Sensor Configuration** (Long-Period Only):
    - Magnetic: Bartington Mag-03 fluxgates (or similar)
      * Full-scale range: ±70,000 nT
      * 24-bit ADC: ±2^23 counts
    - Electric: Non-polarizing Pb-PbCl₂ electrodes
      * Full-scale range: ±100,000 μV / dipole_length
      * 24-bit ADC: ±2^23 counts

**Calibration Formulas** (from Legacy_LP_MT_Process.py):

Magnetic channels (signed, inverted for By):
    Bx [nT] = (chan0 / 2^23 - 1.0) × 70000.0
    Bz [nT] = (chan1 / 2^23 - 1.0) × 70000.0
    By [nT] = -((chan2 / 2^23 - 1.0) × 70000.0)  # Note: inverted

Electric channels (signed, inverted, dipole-normalized):
    Ex [μV/m] = -((chan7 / 2^23 - 1.0) × (100000.0 / dipole_length_ex))
    Ey [μV/m] = -((chan6 / 2^23 - 1.0) × (100000.0 / dipole_length_ey))

**MTH5 Standard Approach**:
    - Store RAW counts (no calibration applied)
    - Create CoefficientFilter objects for each calibration step
    - Set filter.applied = False
    - Matches Phoenix/Zen/NIMS/Metronix/LEMI-423/UoA-PR624 pattern

**Example**:
    >>> from mth5.io.uoa import read_orange
    >>>
    >>> # Read Orange Box data
    >>> run_ts = read_orange(
    ...     '/path/to/site/HFM1-*.BIN',
    ...     station_id='ST61',
    ...     dipole_length_ex=100.0,
    ...     dipole_length_ey=100.0,
    ...     latitude=-31.5,
    ...     longitude=136.5,
    ...     elevation=150.0
    ... )

**References**:
    - Legacy Orange Box processing scripts (University of Adelaide)
    - Flinders University MT deployment procedures
    - ANSIR/AusLAMP long-period MT surveys

Author: Claude Code (Anthropic) with A. Kelbert specifications
Date: 2025-11-07
License: MIT

"""

import logging
from pathlib import Path
from typing import Union, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from mt_metadata.timeseries import Station, Run, Magnetic, Electric
from mt_metadata.timeseries.filters import CoefficientFilter, ChannelResponse
from mth5.timeseries import ChannelTS, RunTS

# Setup module logger
logger = logging.getLogger(__name__ + ".uoa.orange")


# ==============================================================================
# Hardware Calibration Constants
# ==============================================================================

# ADC characteristics (24-bit sigma-delta, unsigned)
ADC_BITS = 24
ADC_MAX_COUNTS = 2 ** 23  # Signed range: ±2^23
ADC_ZERO = 2 ** 23        # Zero point for unsigned → signed conversion

# Bartington fluxgate full-scale range
BARTINGTON_FULL_SCALE_NT = 70000.0  # ±70,000 nT

# Electric field full-scale (before dipole normalization)
ELECTRIC_FULL_SCALE_UV = 100000.0  # ±100,000 μV

# Sample format constants
BYTES_PER_SAMPLE = 18  # Total bytes per complete sample (8 channels + 1 extra)
NCHANNELS = 8


# ==============================================================================
# Calibration Filter Creation Functions
# ==============================================================================

def create_orange_magnetic_filter(component: str, invert: bool = False) -> CoefficientFilter:
    """
    Create calibration filter for Orange Box magnetic channels.

    **Formula**: B [nT] = (counts / 2^23 - 1.0) × 70000.0

    This is a two-step conversion:
        1. Unsigned counts → signed normalized: (counts / 2^23 - 1.0) ∈ [-1, +1)
        2. Normalized → nT: × 70000.0

    :param component: Channel component (hx, hy, hz)
    :type component: str
    :param invert: Whether to invert the signal (True for By/hy)
    :type invert: bool
    :return: Coefficient filter for magnetic calibration
    :rtype: CoefficientFilter

    **Why invert for By**: Legacy Orange Box hardware/orientation convention
    requires By to be inverted for correct geomagnetic coordinate system.
    """
    mag_filter = CoefficientFilter()
    mag_filter.name = f"orange_magnetic_{component}"
    mag_filter.units_in = "counts"
    mag_filter.units_out = "nanotesla"

    # Combined gain: (1 / 2^23) × 70000 × (±1 for invert)
    gain = (BARTINGTON_FULL_SCALE_NT / ADC_MAX_COUNTS)
    if invert:
        gain = -gain

    mag_filter.gain = gain
    mag_filter.offset = -BARTINGTON_FULL_SCALE_NT if not invert else BARTINGTON_FULL_SCALE_NT

    mag_filter.comments = (
        f"Orange Box magnetic calibration: {component.upper()} = "
        f"(counts / 2^23 - 1.0) × {BARTINGTON_FULL_SCALE_NT}"
    )
    if invert:
        mag_filter.comments += " [inverted]"

    return mag_filter


def create_orange_electric_filter(component: str, dipole_length: float) -> CoefficientFilter:
    """
    Create calibration filter for Orange Box electric channels.

    **Formula**: E [μV/m] = -((counts / 2^23 - 1.0) × (100000.0 / dipole_length))

    Three-step conversion:
        1. Unsigned counts → signed normalized: (counts / 2^23 - 1.0) ∈ [-1, +1)
        2. Normalized → μV: × 100000.0
        3. Dipole normalization & inversion: -(μV / dipole_length) → μV/m

    :param component: Channel component (ex, ey)
    :type component: str
    :param dipole_length: Dipole length in meters
    :type dipole_length: float
    :return: Coefficient filter for electric calibration
    :rtype: CoefficientFilter

    **Why invert**: Legacy Orange Box convention inverts electric fields
    for correct polarity in geomagnetic coordinate system.
    """
    elec_filter = CoefficientFilter()
    elec_filter.name = f"orange_electric_{component}_{dipole_length}m"
    elec_filter.units_in = "counts"
    elec_filter.units_out = "microvolts per meter"

    # Combined gain: -(1 / 2^23) × (100000 / dipole_length)
    gain = -(ELECTRIC_FULL_SCALE_UV / ADC_MAX_COUNTS) / dipole_length

    elec_filter.gain = gain
    elec_filter.offset = ELECTRIC_FULL_SCALE_UV / dipole_length  # Positive offset due to negative gain

    elec_filter.comments = (
        f"Orange Box electric calibration: {component.upper()} = "
        f"-((counts / 2^23 - 1.0) × (100000 / {dipole_length}))"
    )

    return elec_filter


# ==============================================================================
# Binary File Reader
# ==============================================================================

class OrangeDataReader:
    """
    Reads a single Orange Box binary file.

    **Binary Format**:
        - 3 header lines (ASCII with \\n terminators)
        - Binary data stream (18 bytes per sample)

    **NEW APPROACH** (following MTH5 standard):
        - Returns RAW counts (no calibration applied)
        - Calibrations described as unapplied filters
        - Matches Phoenix/Zen/NIMS/Metronix/LEMI-423/UoA-PR624 pattern

    :param file_path: Path to .BIN file
    :type file_path: Path or str
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.sample_rate = None
        self.start_time = None
        self.filter_point = None
        self.logger = logger

    def parse_header(self, f) -> dict:
        """
        Parse the 3-line ASCII header.

        :param f: Open binary file handle
        :type f: file object
        :return: Dictionary with sample_rate, start_time, filter_point
        :rtype: dict

        **Header Format**:
            Line 1: " 00008CA0 \\n" (sample rate in hex with leading space)
            Line 2: "Tue Jun 16 02:01:04 2009\\n"
            Line 3: " 7A1" (filter point, 4 bytes hex)

        **Why this format**: Legacy hardware wrote minimal ASCII headers
        for human readability during field operations.
        """
        # Line 1: Sample rate (hex string)
        line1 = f.readline().decode("ascii", errors="ignore").strip()
        self.sample_rate = int(line1, 16)

        # Line 2: Date/time string
        line2 = f.readline().decode("ascii", errors="ignore").strip()
        try:
            self.start_time = pd.to_datetime(line2, format="%a %b %d %H:%M:%S %Y", utc=True)
        except Exception as e:
            self.logger.warning(f"Could not parse start time '{line2}': {e}")
            self.start_time = None

        # Line 3: Filter point (4 bytes, hex)
        filter_bytes = f.read(4).decode("ascii", errors="ignore").strip()
        self.filter_point = int(filter_bytes, 16)

        # Calculate actual sample rate from filter point
        # samples_per_second = 10_000_000 / (512 * filter_point)
        calculated_sr = 10_000_000 / (512 * self.filter_point)

        return {
            "sample_rate": self.sample_rate,
            "start_time": self.start_time,
            "filter_point": self.filter_point,
            "calculated_sample_rate": calculated_sr,
        }

    def read_samples(self, f) -> np.ndarray:
        """
        Read all binary samples from file.

        :param f: Open binary file handle (positioned after header)
        :type f: file object
        :return: Array of shape (n_samples, 8) with raw counts
        :rtype: np.ndarray

        **Binary Layout** (18 bytes per sample, big-endian):
            - Channels 0-2: 3 bytes each (24-bit unsigned)
            - Channels 3-4: 2 bytes each (16-bit unsigned)
            - Channel 5: 1 byte (8-bit unsigned)
            - Channels 6-7: 3 bytes each (24-bit unsigned)
            - 1 extra byte (padding/sync)

        **Why this layout**: Hardware ADC configuration with mixed
        resolution channels for different sensor types.
        """
        samples = []

        while True:
            # Read channels 0-2 (3 bytes each, 24-bit)
            counts = [0] * NCHANNELS

            for i in range(3):
                b1 = f.read(1)
                b2 = f.read(1)
                b3 = f.read(1)
                if not b1 or not b2 or not b3:
                    return np.array(samples, dtype=np.int32) if samples else np.array([], dtype=np.int32).reshape(0, 8)
                counts[i] = (int.from_bytes(b1, byteorder='big') << 16) | \
                           (int.from_bytes(b2, byteorder='big') << 8) | \
                            int.from_bytes(b3, byteorder='big')

            # Read channels 3-4 (2 bytes each, 16-bit)
            for i in range(3, 5):
                b1 = f.read(1)
                b2 = f.read(1)
                if not b1 or not b2:
                    return np.array(samples, dtype=np.int32) if samples else np.array([], dtype=np.int32).reshape(0, 8)
                counts[i] = (int.from_bytes(b1, byteorder='big') << 8) | \
                            int.from_bytes(b2, byteorder='big')

            # Read channel 5 (1 byte, 8-bit)
            b1 = f.read(1)
            if not b1:
                return np.array(samples, dtype=np.int32) if samples else np.array([], dtype=np.int32).reshape(0, 8)
            counts[5] = int.from_bytes(b1, byteorder='big')

            # Read channels 6-7 (3 bytes each, 24-bit)
            for i in range(6, 8):
                b1 = f.read(1)
                b2 = f.read(1)
                b3 = f.read(1)
                if not b1 or not b2 or not b3:
                    return np.array(samples, dtype=np.int32) if samples else np.array([], dtype=np.int32).reshape(0, 8)
                counts[i] = (int.from_bytes(b1, byteorder='big') << 16) | \
                           (int.from_bytes(b2, byteorder='big') << 8) | \
                            int.from_bytes(b3, byteorder='big')

            # Read extra byte (padding/sync)
            extra = f.read(1)
            if not extra:
                return np.array(samples, dtype=np.int32) if samples else np.array([], dtype=np.int32).reshape(0, 8)

            samples.append(counts)

        return np.array(samples, dtype=np.int32)

    def read(self) -> pd.DataFrame:
        """
        Read Orange Box binary file and return RAW counts.

        :return: DataFrame with columns [Bx, Bz, By, Ex, Ey] as RAW counts
        :rtype: pd.DataFrame

        **Note**: Only channels 0, 1, 2, 6, 7 are used (Bx, Bz, By, Ey, Ex).
        Channels 3, 4, 5 are present in file but not used for MT processing.
        """
        with open(self.file_path, 'rb') as f:
            header = self.parse_header(f)
            samples = self.read_samples(f)

        if samples.size == 0:
            self.logger.warning(f"No samples read from {self.file_path}")
            return pd.DataFrame(columns=['Bx', 'Bz', 'By', 'Ex', 'Ey'])

        # Extract MT channels (RAW counts, no calibration)
        # Channel mapping: 0=Bx, 1=Bz, 2=By, 6=Ey, 7=Ex
        df = pd.DataFrame({
            'Bx': samples[:, 0],  # Channel 0
            'Bz': samples[:, 1],  # Channel 1
            'By': samples[:, 2],  # Channel 2
            'Ey': samples[:, 6],  # Channel 6
            'Ex': samples[:, 7],  # Channel 7
        })

        self.logger.info(f"Read {len(df)} samples from {self.file_path.name}")
        return df


# ==============================================================================
# Main Reader Class
# ==============================================================================

class OrangeReader:
    """
    MTH5-compatible reader for Orange Box binary files.

    **NEW APPROACH** (following MTH5 standard):
        - Stores RAW counts in data arrays (no calibrations applied)
        - Describes all calibrations as unapplied filters (filter.applied = False)
        - Matches Phoenix/Zen/NIMS/Metronix/LEMI-423/UoA-PR624 pattern

    :param files: Single file path or list of .BIN files to read
    :type files: str, Path, or list
    :param kwargs: Additional options for metadata
    :type kwargs: dict

    :Keyword Arguments:
        * **station_id** (str) - Station identifier (required)
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 100.0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 100.0)
        * **latitude** (float) - Station latitude in decimal degrees
        * **longitude** (float) - Station longitude in decimal degrees
        * **elevation** (float) - Station elevation in meters

    :Example:
        >>> from mth5.io.uoa import read_orange
        >>> run_ts = read_orange('/path/to/HFM1-*.BIN', station_id='ST61',
        ...                      dipole_length_ex=100.0, dipole_length_ey=100.0)
    """

    def __init__(self, files: Union[str, Path, List[Union[str, Path]]], **kwargs):
        self.files = [Path(f) for f in (files if isinstance(files, list) else [files])]
        self.station_id = kwargs.get('station_id', 'OrangeBox')
        self.dipole_length_ex = kwargs.get('dipole_length_ex', 100.0)
        self.dipole_length_ey = kwargs.get('dipole_length_ey', 100.0)
        self.latitude = kwargs.get('latitude', 0.0)
        self.longitude = kwargs.get('longitude', 0.0)
        self.elevation = kwargs.get('elevation', 0.0)
        self.logger = logger
        self.data = None
        self.header = None
        self.sample_rate = None
        self.start_time = None

    def read(self) -> RunTS:
        """
        Read Orange Box file(s) and return RunTS with channels hx, hy, hz, ex, ey.

        :return: RunTS object containing ChannelTS for each component
        :rtype: RunTS
        """
        # Read all files and concatenate
        dfs = []
        for file_path in self.files:
            reader = OrangeDataReader(file_path)
            df = reader.read()

            # Store header from first file
            if self.header is None:
                self.header = {
                    'sample_rate': reader.sample_rate,
                    'start_time': reader.start_time,
                    'filter_point': reader.filter_point
                }
                self.sample_rate = reader.sample_rate
                self.start_time = reader.start_time

            dfs.append(df)

        if not dfs:
            raise ValueError(f"No data read from files: {self.files}")

        # Concatenate all dataframes
        self.data = pd.concat(dfs, ignore_index=True)

        # Build metadata objects (cached for reuse)
        station_meta = self._build_station_metadata()
        run_meta = self._build_run_metadata()

        # Build ChannelTS objects
        ch_objs = []
        mapping = {
            "Bx": ("hx", "magnetic", 1),
            "By": ("hy", "magnetic", 2),
            "Bz": ("hz", "magnetic", 3),
            "Ex": ("ex", "electric", 4),
            "Ey": ("ey", "electric", 5),
        }

        for src, (code, ch_type, ch_num) in mapping.items():
            if src not in self.data.columns:
                self.logger.warning(f"Channel {src} not found in data")
                continue

            series = self.data[src].to_numpy()
            ch_metadata = self._get_channel_metadata(code, ch_num)

            # Create calibration filters
            filters_list = []

            if ch_type == "magnetic":
                # Magnetic calibration (invert for hy)
                invert = (code == "hy")
                mag_filter = create_orange_magnetic_filter(code, invert=invert)
                filters_list.append(mag_filter)
            else:  # electric
                # Electric calibration
                dipole_length = self.dipole_length_ex if code == "ex" else self.dipole_length_ey
                elec_filter = create_orange_electric_filter(code, dipole_length)
                filters_list.append(elec_filter)

            # Create ChannelResponse
            channel_response = None
            if filters_list:
                channel_response = ChannelResponse(filters_list=filters_list)
                ch_metadata.filter.name = [f.name for f in filters_list]
                ch_metadata.filter.applied = [False] * len(filters_list)

            # Create ChannelTS object
            ch = ChannelTS(
                channel_type=ch_type,
                data=series,
                channel_metadata=ch_metadata,
                run_metadata=run_meta,
                station_metadata=station_meta,
                channel_response=channel_response,
            )

            ch_objs.append(ch)

        return RunTS(
            array_list=ch_objs,
            station_metadata=station_meta,
            run_metadata=run_meta,
        )

    def _build_station_metadata(self) -> Station:
        """Build station metadata object."""
        s = Station()
        s.id = self.station_id
        s.location.latitude = self.latitude
        s.location.longitude = self.longitude
        s.location.elevation = self.elevation
        return s

    def _build_run_metadata(self) -> Run:
        """Build run metadata object."""
        r = Run()
        r.id = "a"  # Default run ID
        r.sample_rate = self.sample_rate if self.sample_rate else 0.0
        r.data_logger.manufacturer = "University of Adelaide"
        r.data_logger.model = "Orange Box"
        r.data_logger.type = "long-period MT"
        r.data_type = "MTLP"
        r.time_period.start = self.start_time.isoformat() if self.start_time else ""
        return r

    def _get_channel_metadata(self, component: str, channel_number: int):
        """Get metadata for a specific channel."""
        if component in ["hx", "hy", "hz"]:
            ch_metadata = Magnetic()
            ch_metadata.type = "magnetic"
            ch_metadata.units = "counts"  # Store RAW counts (MTH5 standard)

            azimuth_map = {"hx": 0, "hy": 90, "hz": 0}
            tilt_map = {"hx": 0, "hy": 0, "hz": 90}
            ch_metadata.measurement_azimuth = azimuth_map.get(component, 0)
            ch_metadata.measurement_tilt = tilt_map.get(component, 0)

            ch_metadata.sensor.manufacturer = "Bartington"
            ch_metadata.sensor.model = "Mag-03"
            ch_metadata.sensor.type = "fluxgate"
        else:
            ch_metadata = Electric()
            ch_metadata.type = "electric"
            ch_metadata.units = "counts"  # Store RAW counts (MTH5 standard)

            azimuth_map = {"ex": 0, "ey": 90}
            ch_metadata.measurement_azimuth = azimuth_map.get(component, 0)
            ch_metadata.measurement_tilt = 0

            if component == "ex":
                ch_metadata.dipole_length = self.dipole_length_ex
            elif component == "ey":
                ch_metadata.dipole_length = self.dipole_length_ey

        ch_metadata.component = component
        ch_metadata.channel_number = channel_number
        ch_metadata.sample_rate = self.sample_rate if self.sample_rate else 0.0

        return ch_metadata


# ==============================================================================
# Convenience Function
# ==============================================================================

def read_orange(
    data_path: Union[str, Path, List[Union[str, Path]]],
    **kwargs
) -> RunTS:
    """
    Read Orange Box binary file(s) and return a RunTS.

    Convenience function to read legacy UoA/Flinders Orange Box long-period
    magnetotelluric binary files with automatic file concatenation.

    :param data_path: Path to .BIN file(s) or list of file paths
    :type data_path: str, Path, or list
    :param kwargs: Additional options (see OrangeReader for details)
    :type kwargs: dict
    :return: RunTS object with channels hx, hy, hz, ex, ey
    :rtype: RunTS

    :Keyword Arguments:
        * **station_id** (str) - Station identifier (required)
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 100.0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 100.0)
        * **latitude** (float) - Station latitude in decimal degrees
        * **longitude** (float) - Station longitude in decimal degrees
        * **elevation** (float) - Station elevation in meters

    :Example:
        >>> from mth5.io.uoa import read_orange
        >>> # Single file
        >>> run_ts = read_orange('HFM1-000.BIN', station_id='ST61')
        >>>
        >>> # Multiple files (glob pattern)
        >>> from glob import glob
        >>> files = sorted(glob('HFM1-*.BIN'))
        >>> run_ts = read_orange(files, station_id='ST61',
        ...                      dipole_length_ex=100.0, dipole_length_ey=100.0)
    """
    # Handle glob patterns
    if isinstance(data_path, (str, Path)):
        data_path = Path(data_path)
        if '*' in str(data_path):
            from glob import glob
            files = sorted(glob(str(data_path)))
            if not files:
                raise ValueError(f"No files found matching pattern: {data_path}")
            data_path = files

    reader = OrangeReader(data_path, **kwargs)
    return reader.read()
