"""
================================================================================
Earth Data Logger PR6-24 Reader for MTH5
================================================================================

Reads ASCII magnetotelluric data from Earth Data PR6-24 data loggers.

The PR6-24 is a 6-channel broadband/long-period MT system that records
data as ASCII text files with one file per channel. Each file contains
voltage samples in microvolts (μV), one value per line.

**File Format**:
    - ASCII text, one sample per line
    - Units: microvolts (μV)
    - No header (raw sample values only)
    - Typical filename: {STATION}YYMMDDHHMMSS.{CHANNEL}
    - Example: EDL_041020030000.BX

**Directory Structures Supported**:
    1. Day folders (001-366): data/294/EDL_041020030000.BX
    2. Flat directory: data/EDL_041020030000.BX
    3. Pre-concatenated: data/EDL_BX.txt (all data in single file)

**Hardware Calibration Details**:

The PR6-24 digitizer has specific hardware gains and voltage dividers:

1. **ADC Characteristics**:
   - 24-bit sigma-delta converter
   - Full-scale range: ±8.388 V
   - Least count: 8.388V / 2^23 ≈ ~1.0 μV
   - Files store: raw μV values from ADC

2. **Bz Voltage Divider**:
   - Hardware: 15kΩ (top) / 10kΩ (bottom) resistor network
   - Reduces ±10V fluxgate output to ±4V at ADC
   - Division ratio: 10/(15+10) = 0.4
   - **Bz files contain the DIVIDED voltage**
   - **Must multiply by 2.5 to recover true sensor output**
   - Why: Keeps Bz in same ±70,000 nT range as Bx/By

3. **Electric Field Terminal Box**:
   - Hardware: Fixed ×10 pre-amplifier
   - Boosts electrode signals before logger
   - **Files contain the AMPLIFIED voltage**
   - **Must divide by 10 to get true electrode potential**
   - Why: Improves SNR over long cable runs

**Sensor Configurations**:

Broadband Mode:
    - Magnetic: LEMI-120 induction coils
    - Electric: Non-polarizing electrodes
    - Frequency response from external .rsp calibration files

Long-Period Mode:
    - Magnetic: Bartington Mag-03 fluxgates
      * Bx, By: 0.007 nT/μV (after removing voltage divider for Bz)
      * Bz: 0.0175 nT/μV = 0.007 nT/μV × 2.5 (includes divider correction)
    - Electric: Non-polarizing electrodes
      * Must divide by terminal box gain (×10) then by dipole length

**Conversion Formulas**:

For Long-Period (Bartington Mag-03):
    Bx [nT] = ASCII_Bx [μV] × 0.007
    By [nT] = ASCII_By [μV] × 0.007
    Bz [nT] = ASCII_Bz [μV] × 0.0175  (includes 2.5× divider correction)

    Ex [mV/m] = ASCII_Ex [μV] / (10 × dipole_length_ex [m])
    Ey [mV/m] = ASCII_Ey [μV] / (10 × dipole_length_ey [m])

For Broadband (LEMI-120):
    - Apply frequency response calibration from .rsp file
    - Units: nT → mV (induction coil output)
    - Terminal box gain (×10) applies to electric only

**Example**:
    >>> from mth5.io.edl import read_uoa
    >>>
    >>> # Long-period with Bartington fluxgates
    >>> run_ts = read_uoa(
    ...     '/path/to/site/data',
    ...     sensor_type='bartington',
    ...     dipole_length_ex=50.0,
    ...     dipole_length_ey=50.0,
    ...     station_id='MT001'
    ... )
    >>>
    >>> # Broadband with LEMI-120 coils
    >>> run_ts = read_uoa(
    ...     '/path/to/site/data',
    ...     sensor_type='lemi120',
    ...     calibration_fn_bx='l120_01.rsp',
    ...     calibration_fn_by='l120_02.rsp',
    ...     calibration_fn_bz='l120_03.rsp',
    ...     dipole_length_ex=50.0,
    ...     dipole_length_ey=50.0
    ... )


Author: 

Date: 2025-11-12
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Optional
import numpy as np
import pandas as pd
from glob import glob

from mt_metadata.timeseries import Station, Run, Magnetic, Electric
from mt_metadata.timeseries.filters import (
    FrequencyResponseTableFilter,
    ChannelResponse,
    CoefficientFilter,
)
from mth5.timeseries import ChannelTS, RunTS

# Setup module logger
logger = logging.getLogger(__name__ + ".uoa.pr624")


# ==============================================================================
# Hardware Calibration Constants
# ==============================================================================

# ADC characteristics (PR6-24 with 24-bit sigma-delta)
ADC_FULL_SCALE_V = 8.388          # ±8.388 V full-scale range
ADC_BITS = 24                      # 24-bit resolution
ADC_COUNTS = 2 ** 23               # Signed range: ±2^23
ADC_LSB_UV = (ADC_FULL_SCALE_V / ADC_COUNTS) * 1e6  # ~1.0 μV per count

# Bz voltage divider (hardware-fixed)
BZ_DIVIDER_R_TOP = 15000.0         # 15 kΩ resistor to sensor
BZ_DIVIDER_R_BOTTOM = 10000.0      # 10 kΩ resistor to ground
BZ_DIVIDER_RATIO = BZ_DIVIDER_R_BOTTOM / (BZ_DIVIDER_R_TOP + BZ_DIVIDER_R_BOTTOM)  # 0.4
BZ_DIVIDER_CORRECTION = 1.0 / BZ_DIVIDER_RATIO  # 2.5 (multiply to recover true voltage)

# Electric field terminal box gain (hardware-fixed)
E_TERMINAL_BOX_GAIN = 10.0         # ×10 pre-amplifier

# Bartington Mag-03 fluxgate conversion (long-period)
# From documentation: ±10V → ±70,000 nT
BARTINGTON_NT_PER_V = 70000.0 / 10.0  # 7000 nT/V
BARTINGTON_NT_PER_UV = BARTINGTON_NT_PER_V / 1e6  # 0.007 nT/μV

# Derived conversion factors for Bartington (long-period)
BARTINGTON_BX_NT_PER_UV = BARTINGTON_NT_PER_UV  # 0.007 nT/μV
BARTINGTON_BY_NT_PER_UV = BARTINGTON_NT_PER_UV  # 0.007 nT/μV
BARTINGTON_BZ_NT_PER_UV = BARTINGTON_NT_PER_UV * BZ_DIVIDER_CORRECTION  # 0.0175 nT/μV

# LEMI-120 induction coil conversion (broadband)
LEMI120_SENSITIVITY_MV_PER_NT = 200.0  # 200 mV/nT at flat response
LEMI120_NT_PER_UV = 0.005  # 0.005 nT/μV = (1000 mV/V) / (200 mV/nT) / (1000 μV/mV)


def read_uoa_coil_response(calibration_fn: Union[str, Path],
                           coil_number: Optional[str] = None) -> FrequencyResponseTableFilter:
    """
    Read LEMI-120 or other coil calibration from .rsp file.

    Auto-detects whether the .rsp file contains normalized or absolute amplitudes:
    - If flat response ~1.0: normalized (nT → nT), needs DC gain filter
    - If flat response ~200: absolute mV/nT (nT → mV), no DC gain needed

    :param calibration_fn: Path to .rsp calibration file
    :type calibration_fn: str or Path
    :param coil_number: Optional coil serial number for identification
    :type coil_number: str, optional
    :return: Frequency response table filter
    :rtype: FrequencyResponseTableFilter

    :Example:
        >>> filter_obj = read_uoa_coil_response("l120_sensor01.rsp", coil_number="01")
        >>> print(f"Normalized: {filter_obj.units_in == filter_obj.units_out}")

    **File Format** (.rsp files):

        Line 1: Channel type ('B' for magnetic)
        Line 2: Column headers (freq, amp, phas)
        Lines 3+: Calibration data (frequency Hz, amplitude, phase degrees)

    .. note::
        Auto-detection uses max amplitude: if > 10, assumes absolute (mV/nT).
        If < 10, assumes normalized (relative to flat response).
    """
    calibration_fn = Path(calibration_fn)

    # Read the file (skip first 2 header lines)
    # Format: frequency (Hz), amplitude, phase (degrees)
    cal_data = np.loadtxt(calibration_fn, skiprows=2)

    # Auto-detect normalization by checking amplitude range
    # Normalized files have flat response ~1.0, absolute files ~100-200
    max_amplitude = np.max(cal_data[:, 1])
    is_normalized = max_amplitude < 10.0

    # Create frequency response filter
    fap = FrequencyResponseTableFilter()
    fap.frequencies = cal_data[:, 0]  # Hz
    fap.amplitudes = cal_data[:, 1]
    fap.phases = np.deg2rad(cal_data[:, 2])  # Convert degrees to radians
    fap.name = f"lemi_120_{coil_number}_response" if coil_number else "lemi_120_response"
    fap.type = "frequency response table"
    fap.calibration_date = "1970-01-01T00:00:00+00:00"  # Default

    if is_normalized:
        # Normalized: nT → nT (relative correction)
        fap.units_in = "nanotesla"
        fap.units_out = "nanotesla"
        fap.comments = f"LEMI-120 normalized response from {calibration_fn.name}"
    else:
        # Absolute: nT → mV (includes DC sensitivity)
        fap.units_in = "nanotesla"
        fap.units_out = "millivolts"
        fap.comments = f"LEMI-120 absolute response (mV/nT) from {calibration_fn.name}"

    return fap


def create_bz_divider_filter() -> CoefficientFilter:
    """
    Create filter for Bz voltage divider correction (Bartington mode only).

    **Hardware**: 15kΩ/10kΩ voltage divider reduces Bz sensor output.
    **Files contain**: Divided voltage (×0.4)
    **Filter applies**: Multiply by 2.5 to recover true sensor voltage

    :return: Coefficient filter for Bz divider correction
    :rtype: CoefficientFilter

    **Why this filter**:
    The Bartington Mag-03 Bz fluxgate outputs ±10V for ±70,000 nT.
    A hardware voltage divider (15kΩ top / 10kΩ bottom) reduces this to
    ±4V at the ADC input. The ASCII files contain the divided voltage.
    To get the actual sensor output, multiply by 2.5.

    .. note:: Only applies to Bartington mode. LEMI-120 coils have no divider.
    """
    bz_filter = CoefficientFilter()
    bz_filter.name = "uoa_bz_voltage_divider"
    bz_filter.units_in = "microvolts"
    bz_filter.units_out = "microvolts"
    bz_filter.gain = BZ_DIVIDER_CORRECTION  # 2.5
    bz_filter.comments = (
        "Bz voltage divider correction: 15kΩ/10kΩ = 0.4 ratio. "
        "Multiply by 2.5 to recover true sensor voltage. "
        "Only applies to Bartington Mag-03 fluxgate (long-period mode)."
    )
    return bz_filter


def create_efield_gain_filter() -> CoefficientFilter:
    """
    Create filter for E-field terminal box gain removal.

    **Hardware**: Fixed ×10 pre-amplifier in electric field terminal box
    **Files contain**: Amplified voltage (×10)
    **Filter applies**: Divide by 10 to recover true electrode potential

    :return: Coefficient filter for E-field gain removal
    :rtype: CoefficientFilter

    **Why this filter**:
    The E-field terminal box contains a fixed ×10 pre-amplifier to boost
    the electrode signals before transmission over long cables. This improves
    SNR. The ASCII files contain the amplified voltage. To get the actual
    electrode potential, divide by 10.
    """
    efield_filter = CoefficientFilter()
    efield_filter.name = "uoa_efield_terminal_box_gain"
    efield_filter.units_in = "microvolts"
    efield_filter.units_out = "microvolts"
    efield_filter.gain = 1.0 / E_TERMINAL_BOX_GAIN  # 0.1
    efield_filter.comments = (
        "E-field terminal box gain removal: ×10 hardware pre-amplifier. "
        "Divide by 10 to recover true electrode potential."
    )
    return efield_filter


def create_lemi120_dc_gain_filter(component: str) -> CoefficientFilter:
    """
    Create LEMI-120 DC gain filter: microvolts → nanotesla.

    Applies DC sensitivity of 200 mV/nT = 0.005 nT/μV.
    Only used when .rsp file contains normalized response (nT → nT).

    :param component: Component name ('hx', 'hy', or 'hz')
    :type component: str
    :return: Coefficient filter for DC gain
    :rtype: CoefficientFilter
    """
    dc_filter = CoefficientFilter()
    dc_filter.name = f"lemi120_dc_gain_{component}"
    dc_filter.units_in = "microvolts"
    dc_filter.units_out = "nanotesla"
    dc_filter.gain = LEMI120_NT_PER_UV  # 0.005 nT/μV
    dc_filter.comments = "LEMI-120 DC sensitivity: 200 mV/nT = 0.005 nT/μV"
    return dc_filter


def create_bartington_calibration_filter(component: str) -> CoefficientFilter:
    """
    Create Bartington Mag-03 fluxgate calibration filter.

    Converts microvolts to nanotesla using manufacturer specifications.

    :param component: Component name ('hx', 'hy', or 'hz')
    :type component: str
    :return: Coefficient filter for Bartington calibration
    :rtype: CoefficientFilter

    **Bartington Mag-03 Specifications**:
    - Range: ±70,000 nT
    - Output: ±10 V
    - Conversion: 7000 nT/V = 0.007 nT/μV

    **Important**: For Bz, this filter should be applied AFTER the voltage
    divider correction filter, as it expects the true sensor voltage.

    .. note:: This is a linear conversion, frequency-independent.
    """
    if component == 'hz':
        # For Bz: expects voltage AFTER divider correction
        gain = BARTINGTON_NT_PER_UV  # 0.007 nT/μV
        comments = (
            "Bartington Mag-03 Bz calibration: 0.007 nT/μV. "
            "Applied after voltage divider correction. "
            "±10V → ±70,000 nT (7000 nT/V)."
        )
    else:
        # For Bx, By: direct conversion
        gain = BARTINGTON_NT_PER_UV  # 0.007 nT/μV
        comments = (
            f"Bartington Mag-03 {component.upper()} calibration: 0.007 nT/μV. "
            "±10V → ±70,000 nT (7000 nT/V)."
        )

    bart_filter = CoefficientFilter()
    bart_filter.name = f"uoa_bartington_{component}"
    bart_filter.units_in = "microvolts"
    bart_filter.units_out = "nanotesla"
    bart_filter.gain = gain
    bart_filter.comments = comments
    return bart_filter


def create_dipole_length_filter(component: str, dipole_length: float) -> CoefficientFilter:
    """
    Create dipole length normalization filter for electric fields.

    Converts microvolts to electric field (mV/km) using dipole length.

    :param component: Component name ('ex' or 'ey')
    :type component: str
    :param dipole_length: Dipole length in meters
    :type dipole_length: float
    :return: Coefficient filter for dipole normalization
    :rtype: CoefficientFilter

    **Conversion**:
    E [mV/km] = V [μV] / (dipole_length [m] / 1000)
              = V [μV] × (1000 / dipole_length)

    .. note:: This filter should be applied AFTER the terminal box gain filter.
    """
    if dipole_length <= 0:
        dipole_length = 1.0  # Fallback

    dipole_filter = CoefficientFilter()
    dipole_filter.name = f"uoa_dipole_{component}_{dipole_length:.1f}m"
    dipole_filter.units_in = "microvolts"
    dipole_filter.units_out = "millivolts per kilometer"
    dipole_filter.gain = 1000.0 / dipole_length  # Converts μV to mV/km
    dipole_filter.comments = (
        f"Dipole length normalization: {dipole_length}m. "
        f"Converts electrode potential to electric field."
    )
    return dipole_filter


# ==============================================================================
# EDL Data File Reader
# ==============================================================================

class UoADataReader:
    """
    Reads EDL ASCII data files for a single channel.

    Handles multiple file organization patterns:
    - Day-numbered folders (001-366)
    - Flat directory structure
    - Pre-concatenated single files

    **Why**: EDL data can be organized differently depending on whether it's
    fresh from the logger, copied for transport, or pre-processed. This class
    abstracts those differences.

    :param channel: Channel name (BX, BY, BZ, EX, EY, TP, etc.)
    :type channel: str
    :param data_path: Path to data directory or specific file
    :type data_path: Path
    :param station_prefix: Station identifier prefix (e.g., 'EDL_', 'MT001_')
    :type station_prefix: str, optional

    :Example:
        >>> reader = UoADataReader('BX', Path('/data/site1'), station_prefix='EDL_')
        >>> data = reader.read()
        >>> print(f"Read {len(data)} samples")
    """

    def __init__(self, channel: str, data_path: Path, station_prefix: Optional[str] = None):
        self.channel = channel.upper()
        self.data_path = Path(data_path)
        self.station_prefix = station_prefix or ''
        self.logger = logger

    def find_files(self) -> List[Path]:
        """
        Find all data files for this channel.

        Search patterns (in order):
        1. {station_prefix}YYMMDDHHMMSS.{channel} in day folders
        2. {station_prefix}*.{channel} in data_path
        3. *.{channel} (any file ending in .BX, .BY, etc.)

        :return: List of file paths sorted chronologically
        :rtype: list of Path

        **Why these patterns**:
        - Pattern 1: Fresh from logger with day folders (standard EDL format)
        - Pattern 2: Copied to flat directory with station prefix
        - Pattern 3: Pre-concatenated or renamed files without station prefix
        """
        # Pattern 1: Look in day-numbered subdirectories (001-366)
        if self.data_path.is_dir():
            # Try day folders first
            day_pattern = f"**/{self.station_prefix}*.{self.channel}"
            day_files = sorted(self.data_path.glob(day_pattern))
            if day_files:
                self.logger.info(f"Found {len(day_files)} files for {self.channel} in day folders")
                return day_files

            # Pattern 2: Flat directory with timestamped files
            flat_pattern = f"{self.station_prefix}*.{self.channel}"
            flat_files = sorted(self.data_path.glob(flat_pattern))
            if flat_files:
                self.logger.info(f"Found {len(flat_files)} files for {self.channel} in flat directory")
                return flat_files

            # Pattern 3: Pre-concatenated files (no station prefix)
            # Simply match any file ending in .{channel} (e.g., *.BX, *.BY)
            no_prefix_pattern = f"*.{self.channel}"
            no_prefix_files = sorted(self.data_path.glob(no_prefix_pattern))
            if no_prefix_files:
                self.logger.info(f"Found {len(no_prefix_files)} file(s) for {self.channel} without station prefix")
                return no_prefix_files

        # If data_path is a file, use it directly
        elif self.data_path.is_file():
            self.logger.info(f"Using specified file for {self.channel}: {self.data_path.name}")
            return [self.data_path]

        self.logger.warning(f"No files found for channel {self.channel} in {self.data_path}")
        return []

    def read(self) -> np.ndarray:
        """
        Read and concatenate all files for this channel.

        :return: Array of float values in microvolts (μV)
        :rtype: np.ndarray

        **File Format**:
            - ASCII text, one sample per line
            - Values in microvolts (μV)
            - No header
            - Lines with non-numeric content are skipped

        **Why ASCII**: The PR6-24 outputs ASCII for simplicity and
        field-readability, despite higher disk usage than binary formats.
        """
        files = self.find_files()
        if not files:
            self.logger.error(f"No data files found for channel {self.channel}")
            return np.array([])

        all_data = []
        for file_path in files:
            try:
                # Read ASCII file: one float per line
                # Use numpy for speed, skip invalid lines
                data = np.loadtxt(file_path, dtype=float, comments=None)

                # Handle both 1D array (single column) and 2D array (if accidentally multi-column)
                if data.ndim > 1:
                    data = data.flatten()

                all_data.append(data)
                self.logger.debug(f"Read {len(data)} samples from {file_path.name}")

            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue

        if not all_data:
            return np.array([])

        # Concatenate all data
        combined = np.concatenate(all_data)
        self.logger.info(f"Channel {self.channel}: {len(combined)} total samples from {len(files)} file(s)")

        return combined

# ==============================================================================
# Main EDL Reader Class
# ==============================================================================

class UoAReader:
    """
    MTH5-compatible reader for Earth Data PR6-24 ASCII data files.

    Reads EDL magnetotelluric data with handling of hardware calibrations:
    - Bz voltage divider (15kΩ/10kΩ) correction
    - Electric field terminal box gain (×10) removal
    - Bartington fluxgate or LEMI-120 coil calibration

    **File Discovery**:
        - Day-numbered folders (001-366)
        - Flat directories
        - Pre-concatenated files
        - Works without recorder.ini or GPS files

    :param data_path: Path to data directory or file
    :type data_path: str or Path
    :param sensor_type: 'bartington' or 'lemi120' (default: 'bartington')
    :type sensor_type: str, optional
    :param kwargs: Additional parameters (see below)
    :type kwargs: dict

    :Keyword Arguments:
        * **sample_rate** (float) - Sample rate in Hz (required if not auto-detected)
        * **station_id** (str) - Station identifier (default: auto-detect from path)
        * **station_prefix** (str) - File prefix like 'EDL_', 'MT001_' (default: '')
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 1.0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 1.0)
        * **calibration_fn_bx** (str) - LEMI-120 .rsp file for Bx (if lemi120 mode)
        * **calibration_fn_by** (str) - LEMI-120 .rsp file for By (if lemi120 mode)
        * **calibration_fn_bz** (str) - LEMI-120 .rsp file for Bz (if lemi120 mode)
        * **latitude** (float) - Station latitude in decimal degrees
        * **longitude** (float) - Station longitude in decimal degrees
        * **elevation** (float) - Station elevation in meters

    :Example:
        >>> # Long-period with Bartington
        >>> reader = UoAReader('/data/site1', sensor_type='bartington',
        ...                     sample_rate=10.0,
        ...                     dipole_length_ex=50.0,
        ...                     dipole_length_ey=50.0)
        >>> run_ts = reader.read()
        >>>
        >>> # Broadband with LEMI-120
        >>> reader = UoAReader('/data/site1', sensor_type='lemi120',
        ...                     sample_rate=500.0,
        ...                     calibration_fn_bx='l120_01.rsp',
        ...                     calibration_fn_by='l120_02.rsp',
        ...                     calibration_fn_bz='l120_03.rsp',
        ...                     dipole_length_ex=50.0,
        ...                     dipole_length_ey=50.0)
        >>> run_ts = reader.read()

    .. note::
        - Sample rate MUST be provided (no automatic detection from ASCII files)
        - Bz data is automatically corrected for voltage divider
        - Ex/Ey data is automatically corrected for terminal box gain
        - For accurate E-field, provide actual dipole lengths in meters
    """

    def __init__(self, data_path: Union[str, Path], sensor_type: str = 'bartington', **kwargs):
        self.logger = logger
        self.data_path = Path(data_path)
        self.sensor_type = sensor_type.lower()

        # Required parameters
        self.sample_rate = kwargs.get('sample_rate', None)
        if self.sample_rate is None:
            raise ValueError("sample_rate parameter is required (EDL ASCII files don't contain this info)")

        # Station identification
        self.station_id = kwargs.get('station_id', None)
        self.station_prefix = kwargs.get('station_prefix', '')

        # Dipole lengths for electric field conversion
        self.dipole_length_ex = kwargs.get('dipole_length_ex', 1.0)
        self.dipole_length_ey = kwargs.get('dipole_length_ey', 1.0)

        # Calibration files (LEMI-120 mode only)
        self.calibration_fn_bx = kwargs.get('calibration_fn_bx', None)
        self.calibration_fn_by = kwargs.get('calibration_fn_by', None)
        self.calibration_fn_bz = kwargs.get('calibration_fn_bz', None)

        # Optional location metadata
        self.latitude = kwargs.get('latitude', None)
        self.longitude = kwargs.get('longitude', None)
        self.elevation = kwargs.get('elevation', None)

        # Data storage
        self.data = None
        self.n_samples = 0

    def _get_station_id(self) -> str:
        """
        Determine station ID from kwargs or path.

        Priority:
        1. Explicit station_id parameter
        2. Parent directory name

        :return: Station identifier
        :rtype: str
        """
        if self.station_id:
            return self.station_id

        # Use parent directory name
        return self.data_path.parent.name if self.data_path.is_file() else self.data_path.name

    def read(self) -> RunTS:
        """
        Read EDL data files and return a RunTS object.

        - Stores RAW microvolts in data array
        - Describes all calibrations as filters with filter.applied = False

        Process:
        1. Find and read all 5 MT channels (BX, BY, BZ, EX, EY)
        2. Store RAW microvolts
        3. Build calibration filters based on sensor type
        4. Build metadata objects
        5. Create ChannelTS objects with filters
        6. Return RunTS with all channels

        :return: RunTS object containing ChannelTS for bx, by, bz, ex, ey
        :rtype: RunTS

        **Calibration Filters Created**:

        For Bartington (long-period):
            - Bz: voltage divider filter (×2.5), then Bartington filter (0.007 nT/μV)
            - Bx, By: Bartington filter (0.007 nT/μV)
            - Ex, Ey: terminal box gain filter (÷10), then dipole filter

        For LEMI-120 (broadband):
            - Bx, By, Bz: LEMI-120 frequency response from .rsp files
            - Ex, Ey: terminal box gain filter (÷10), then dipole filter
        """
        self.logger.info(f"Reading EDL data from {self.data_path}")
        self.logger.info(f"Sensor type: {self.sensor_type}, Sample rate: {self.sample_rate} Hz")

        # Define MT channels
        channels = ['BX', 'BY', 'BZ', 'EX', 'EY']
        channel_data = {}

        # Read each channel
        for channel in channels:
            reader = UoADataReader(channel, self.data_path, self.station_prefix)
            data = reader.read()

            if len(data) == 0:
                self.logger.warning(f"No data found for channel {channel}")
                continue

            channel_data[channel] = data

        # Check we have data
        if not channel_data:
            raise ValueError(f"No channel data found in {self.data_path}")

        # Find minimum length (trim all to same length)
        min_length = min(len(data) for data in channel_data.values())
        self.n_samples = min_length

        for channel in channel_data:
            channel_data[channel] = channel_data[channel][:min_length]

        self.logger.info(f"Read {min_length} samples across {len(channel_data)} channels")

        # Build metadata objects ONCE (reuse pattern from LEMI-423)
        station_meta = self._build_station_metadata()
        run_meta = self._build_run_metadata()

        # Build ChannelTS objects with RAW data and calibration filters
        ch_objs: List[ChannelTS] = []

        # Mapping: file channel name → (MTH5 code, type, channel_number)
        mapping = {
            "BX": ("hx", "magnetic", 0),
            "BY": ("hy", "magnetic", 1),
            "BZ": ("hz", "magnetic", 2),
            "EX": ("ex", "electric", 3),
            "EY": ("ey", "electric", 4),
        }

        for src, (code, ch_type, ch_num) in mapping.items():
            if src not in channel_data:
                self.logger.warning(f"Channel {src} not found in data")
                continue

            # Get RAW data in microvolts from ASCII file
            raw_uv = channel_data[src]

            # Build channel metadata
            ch_metadata = self._get_channel_metadata(code, ch_num)

            # Create calibration filters (NOT applied to data)
            if ch_type == "magnetic":
                channel_response = self._create_magnetic_filters(code)
            else:  # electric
                channel_response = self._create_electric_filters(code)

            # Update metadata to reference filters
            if channel_response is not None:
                ch_metadata.filter.name = [filt.name for filt in channel_response.filters_list]
                ch_metadata.filter.applied = [False] * len(channel_response.filters_list)

            # Create ChannelTS object with RAW data (microvolts)
            ch = ChannelTS(
                channel_type=ch_type,
                data=raw_uv,  # Store RAW microvolts (NOT calibrated)
                channel_metadata=ch_metadata,
                run_metadata=run_meta,
                station_metadata=station_meta,
                channel_response=channel_response,
            )

            ch_objs.append(ch)

        # Return RunTS with all channels
        return RunTS(
            array_list=ch_objs,
            station_metadata=station_meta,
            run_metadata=run_meta,
        )

    def _create_magnetic_filters(self, component: str) -> Optional[ChannelResponse]:
        """
        Create calibration filters for magnetic channels (NOT applied to data).

        Following MTH5 standard: data stays as raw microvolts, filters describe
        the calibration chain.

        :param component: Component name ('hx', 'hy', or 'hz')
        :type component: str
        :return: ChannelResponse with calibration filters, or None
        :rtype: ChannelResponse or None

        **Filter chains created**:

        Bartington mode (long-period):
            - Bz: [voltage_divider_filter, bartington_filter]
            - Bx, By: [bartington_filter]

        LEMI-120 mode (broadband):
            - Normalized .rsp (nT→nT): [dc_gain_filter, frequency_response]
            - Absolute .rsp (nT→mV): [frequency_response]
        """
        filters = []

        if self.sensor_type == 'bartington':
            # Bartington Mag-03 fluxgate (long-period)
            if component == 'hz':
                # Bz: needs voltage divider correction first
                filters.append(create_bz_divider_filter())

            # Then Bartington calibration for all components
            filters.append(create_bartington_calibration_filter(component))

        elif self.sensor_type == 'lemi120':
            # LEMI-120 induction coils (broadband)
            # Load frequency response from .rsp file
            cal_file_map = {
                'hx': self.calibration_fn_bx,
                'hy': self.calibration_fn_by,
                'hz': self.calibration_fn_bz,
            }

            cal_fn = cal_file_map.get(component)
            if cal_fn is None:
                self.logger.warning(f"No calibration file specified for {component}")
                return None

            try:
                coil_filter = read_uoa_coil_response(cal_fn, coil_number=component)

                # If .rsp file is normalized (nT → nT), need DC gain filter first
                if coil_filter.units_in == coil_filter.units_out == "nanotesla":
                    filters.append(create_lemi120_dc_gain_filter(component))

                filters.append(coil_filter)
            except Exception as e:
                self.logger.error(f"Error reading calibration file {cal_fn}: {e}")
                return None
        else:
            raise ValueError(f"Unknown sensor type: {self.sensor_type}")

        if filters:
            return ChannelResponse(filters_list=filters)
        return None

    def _create_electric_filters(self, component: str) -> Optional[ChannelResponse]:
        """
        Create calibration filters for electric channels (NOT applied to data).

        Following MTH5 standard: data stays as raw microvolts, filters describe
        the calibration chain.

        :param component: Component name ('ex' or 'ey')
        :type component: str
        :return: ChannelResponse with calibration filters
        :rtype: ChannelResponse

        **Filter chain created**:
        [terminal_box_gain_filter, dipole_length_filter]

        **Why these filters**:
        1. Terminal box gain filter: Removes ×10 hardware pre-amplifier gain
        2. Dipole length filter: Normalizes by dipole length to get E-field

        Final conversion: μV → μV (÷10) → mV/km
        """
        filters = []

        # Filter 1: Remove terminal box gain (÷10)
        filters.append(create_efield_gain_filter())

        # Filter 2: Normalize by dipole length
        dipole_length = self.dipole_length_ex if component == 'ex' else self.dipole_length_ey
        if dipole_length <= 0:
            self.logger.warning(f"Dipole length for {component} is {dipole_length}m, using 1m")
            dipole_length = 1.0

        filters.append(create_dipole_length_filter(component, dipole_length))

        return ChannelResponse(filters_list=filters)

    def _build_station_metadata(self) -> Station:
        """
        Build Station metadata object.

        Uses provided location or defaults to zeros.

        :return: Station metadata
        :rtype: Station
        """
        s = Station()
        s.id = self._get_station_id()

        # Location (use provided or default to 0,0,0)
        s.location.latitude = self.latitude if self.latitude is not None else 0.0
        s.location.longitude = self.longitude if self.longitude is not None else 0.0
        s.location.elevation = self.elevation if self.elevation is not None else 0.0

        s.location.declination.value = 0.0  # User should update if known
        s.geographic_name = "Unknown"  # User should update

        return s

    def _build_run_metadata(self) -> Run:
        """
        Build Run metadata object.

        :return: Run metadata
        :rtype: Run
        """
        r = Run()
        r.id = "a"  # Default run ID
        r.sample_rate = self.sample_rate

        # Data logger information
        r.data_logger.model = "PR6-24"
        r.data_logger.manufacturer = "Earth Data"
        r.data_logger.type = "broadband" if self.sensor_type == 'lemi120' else "long period"

        # Data type
        r.data_type = "MTBB" if self.sensor_type == 'lemi120' else "MTLP"

        return r

    def _get_channel_metadata(self, component: str, channel_number: int):
        """
        Build channel metadata object.

        :param component: Component name ('hx', 'hy', 'hz', 'ex', 'ey')
        :type component: str
        :param channel_number: Channel number (0-4)
        :type channel_number: int
        :return: Channel metadata (Magnetic or Electric)
        :rtype: Magnetic or Electric
        """
        if component in ["hx", "hy", "hz"]:
            ch_metadata = Magnetic()
            ch_metadata.type = "magnetic"
            # Store RAW microvolts (following MTH5 standard like Zen, Phoenix, NIMS)
            ch_metadata.units = "microvolts"

            # Sensor information
            ch_metadata.sensor.manufacturer = "Bartington" if self.sensor_type == 'bartington' else "LEMI"
            ch_metadata.sensor.model = "Mag-03" if self.sensor_type == 'bartington' else "LEMI-120"
            ch_metadata.sensor.type = "fluxgate" if self.sensor_type == 'bartington' else "induction coil"

        else:  # electric
            ch_metadata = Electric()
            ch_metadata.type = "electric"
            # Store RAW microvolts (following MTH5 standard)
            ch_metadata.units = "microvolts"

            # Store dipole length in metadata for filter application
            if component == "ex":
                ch_metadata.dipole_length = self.dipole_length_ex
                ch_metadata.measurement_azimuth = 0.0  # Default N-S
            else:  # ey
                ch_metadata.dipole_length = self.dipole_length_ey
                ch_metadata.measurement_azimuth = 90.0  # Default E-W

            ch_metadata.measurement_tilt = 0.0

        ch_metadata.component = component
        ch_metadata.channel_number = channel_number
        ch_metadata.sample_rate = self.sample_rate

        return ch_metadata


# ==============================================================================
# Convenience Function (Entry Point)
# ==============================================================================

def read_uoa(data_path: Union[str, Path], **kwargs) -> RunTS:
    """
    Read Earth Data PR6-24 ASCII data files and return a RunTS.

    Convenience function for reading EDL magnetotelluric data with automatic
    handling of hardware calibrations (Bz voltage divider, E-field terminal box gain).

    :param data_path: Path to data directory or file
    :type data_path: str or Path
    :param kwargs: Additional parameters (see below)
    :type kwargs: dict

    :Keyword Arguments:
        * **sensor_type** (str) - 'bartington' or 'lemi120' (default: 'bartington')
        * **sample_rate** (float) - Sample rate in Hz (REQUIRED)
        * **station_id** (str) - Station identifier
        * **station_prefix** (str) - File prefix like 'EDL_', 'MT001_'
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 1.0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 1.0)
        * **calibration_fn_bx** (str) - LEMI-120 .rsp file for Bx (broadband mode)
        * **calibration_fn_by** (str) - LEMI-120 .rsp file for By (broadband mode)
        * **calibration_fn_bz** (str) - LEMI-120 .rsp file for Bz (broadband mode)
        * **latitude** (float) - Station latitude in decimal degrees
        * **longitude** (float) - Station longitude in decimal degrees
        * **elevation** (float) - Station elevation in meters

    :return: RunTS object containing ChannelTS for hx, hy, hz, ex, ey
    :rtype: RunTS

    :Example:
        >>> from mth5.io.edl import read_uoa
        >>>
        >>> # Long-period with Bartington fluxgates
        >>> run_ts = read_uoa(
        ...     '/data/site1',
        ...     sensor_type='bartington',
        ...     sample_rate=10.0,
        ...     dipole_length_ex=50.0,
        ...     dipole_length_ey=50.0,
        ...     station_id='MT001'
        ... )
        >>>
        >>> # Broadband with LEMI-120 induction coils
        >>> run_ts = read_uoa(
        ...     '/data/site2',
        ...     sensor_type='lemi120',
        ...     sample_rate=500.0,
        ...     calibration_fn_bx='/path/to/l120_01.rsp',
        ...     calibration_fn_by='/path/to/l120_02.rsp',
        ...     calibration_fn_bz='/path/to/l120_03.rsp',
        ...     dipole_length_ex=50.0,
        ...     dipole_length_ey=50.0
        ... )
        >>>
        >>> # Access channel data
        >>> hx_data = run_ts.hx.to_numpy()
        >>> print(f"Station: {run_ts.station_metadata.id}")

    **Notes**:

    1. **Sample Rate** (REQUIRED):
       ASCII files don't contain sample rate - you must provide it

    2. **Hardware Calibrations** (automatic):
       - Bz voltage divider: 15kΩ/10kΩ = 0.4 (data multiplied by 2.5)
       - E-field terminal box: ×10 gain (data divided by 10)

    3. **Bartington Mode** (long-period, ~10 Hz):
       - Bx, By: 0.007 nT/μV
       - Bz: 0.0175 nT/μV (includes voltage divider)
       - Output units: nanotesla (nT)

    4. **LEMI-120 Mode** (broadband, ~100-500 Hz):
       - Requires calibration .rsp files
       - Frequency response stored but not applied (filter.applied = False)
       - Output units: millivolts (mV)

    5. **Electric Fields**:
       - Always corrected for terminal box gain (÷10)
       - Normalized by dipole length
       - Output units: mV/m

    .. note::
        File organization is flexible - works with day folders, flat directories,
        or pre-concatenated files.

    .. seealso::
        :class:`UoAReader` - Main reader class
        :func:`read_uoa_coil_response` - Read LEMI-120 calibration files
    """
    reader = UoAReader(data_path, **kwargs)
    return reader.read()
