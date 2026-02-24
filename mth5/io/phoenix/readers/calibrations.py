# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:21:35 2023

@author: jpeacock

Calibrations can come in json files.  the JSON file includes filters
for all lowpass filters, so you need to match the lowpass filter used in the
setup with the lowpass filter.  Then you need to add the dipole length and
sensor calibrations.
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

from .helpers import read_json_to_object


if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================


class PhoenixCalibration:
    """
    Phoenix Geophysics calibration data reader and filter manager.

    This class reads Phoenix calibration files in JSON format and provides
    access to frequency response filters for different channels and lowpass
    filter settings. It supports both receiver and sensor calibration files.

    Parameters
    ----------
    cal_fn : str or pathlib.Path, optional
        Path to the calibration file to read. If provided, the file will be
        loaded automatically during initialization.
    **kwargs : Any
        Additional keyword arguments that will be set as instance attributes.

    Attributes
    ----------
    obj : Any or None
        The parsed calibration object containing all calibration data.
    """

    def __init__(self, cal_fn: str | Path | None = None, **kwargs: Any) -> None:
        self.obj: Any = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.cal_fn = cal_fn

    def __str__(self) -> str:
        """String representation of PhoenixCalibration."""
        lines = ["Phoenix Response Filters"]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Detailed string representation of PhoenixCalibration."""
        return self.__str__()

    @property
    def cal_fn(self) -> Path:
        """
        Path to the calibration file.

        Returns
        -------
        pathlib.Path
            The path to the calibration file.
        """
        return self._cal_fn

    @cal_fn.setter
    def cal_fn(self, cal_fn: str | Path | None) -> None:
        """
        Set the calibration file path and automatically read the file.

        Parameters
        ----------
        cal_fn : str, pathlib.Path, or None
            Path to the calibration file. If None, no action is taken.
            If the file exists, it will be read automatically.

        Raises
        ------
        IOError
            If the specified file does not exist.
        """
        if cal_fn is not None:
            self._cal_fn = Path(cal_fn)
            if self._cal_fn.exists():
                self.read()
            else:
                raise IOError(f"Could not find file {cal_fn}")

    @property
    def calibration_date(self) -> MTime | None:
        """
        Get the calibration date from the loaded calibration data.

        Returns
        -------
        MTime or None
            The calibration date as an MTime object, or None if no data is loaded.
        """
        if self._has_read():
            return MTime(time_stamp=self.obj.timestamp_utc)
        return None

    def _has_read(self) -> bool:
        """
        Check if calibration data has been loaded.

        Returns
        -------
        bool
            True if calibration data is loaded, False otherwise.
        """
        return self.obj is not None

    def get_max_freq(
        self, freq: NDArray[np.floating] | list[float] | np.ndarray
    ) -> int:
        """
        Calculate the maximum frequency for filter naming.

        Determines the power-of-10 frequency limit based on the maximum
        frequency in the input array. Used to name filters as
        {channel}_{max_freq}hz_lowpass.

        Parameters
        ----------
        freq : numpy.ndarray
            Array of frequency values in Hz.

        Returns
        -------
        int
            The power-of-10 frequency limit (e.g., 1000 for frequencies up to 9999 Hz).

        Examples
        --------
        >>> cal = PhoenixCalibration()
        >>> freq = np.array([1.0, 10.0, 100.0, 1500.0])
        >>> cal.get_max_freq(freq)
        1000
        """
        return int(10 ** np.floor(np.log10(np.array(freq).max())))

    @property
    def base_filter_name(self) -> str | None:
        """
        Generate the base filter name from instrument information.

        Creates a standardized filter name prefix based on the instrument
        type, model, and serial number from the calibration data.

        Returns
        -------
        str or None
            Base filter name in format "{instrument_type}_{instrument_model}_{serial}"
            converted to lowercase, or None if no data is loaded.

        Examples
        --------
        >>> cal = PhoenixCalibration("calibration.json")
        >>> cal.base_filter_name
        'mtu-5c_rmt03-j_666'
        """
        if self._has_read():
            return (
                f"{self.obj.instrument_type}_"
                f"{self.obj.instrument_model}_"
                f"{self.obj.inst_serial}"
            ).lower()
        return None

    def get_filter_lp_name(self, channel: str, max_freq: int) -> str:
        """
        Generate a lowpass filter name for a specific channel and frequency.

        Creates a standardized filter name for receiver calibration filters
        in the format: {base_filter_name}_{channel}_{max_freq}hz_lowpass

        Parameters
        ----------
        channel : str
            Channel identifier (e.g., 'e1', 'h2').
        max_freq : int
            Maximum frequency in Hz for the lowpass filter.

        Returns
        -------
        str
            Complete lowpass filter name in lowercase.

        Examples
        --------
        >>> cal = PhoenixCalibration("calibration.json")
        >>> cal.get_filter_lp_name("e1", 1000)
        'mtu-5c_rmt03-j_666_e1_1000hz_lowpass'
        """
        return f"{self.base_filter_name}_{channel}_{max_freq}hz_lowpass".lower()

    def get_filter_sensor_name(self, sensor: str) -> str:
        """
        Generate a sensor filter name for a specific sensor.

        Creates a standardized filter name for sensor calibration filters
        in the format: {base_filter_name}_{sensor}

        Parameters
        ----------
        sensor : str
            Sensor identifier or serial number.

        Returns
        -------
        str
            Complete sensor filter name in lowercase.

        Examples
        --------
        >>> cal = PhoenixCalibration("calibration.json")
        >>> cal.get_filter_sensor_name("sensor123")
        'mtu-5c_rmt03-j_666_sensor123'
        """
        return f"{self.base_filter_name}_{sensor}".lower()

    def read(self, cal_fn: str | Path | None = None) -> None:
        """
        Read and parse a Phoenix calibration file.

        Loads calibration data from a JSON file and creates frequency response
        filters for each channel and frequency band. The method creates channel
        attributes (e.g., self.e1, self.h2) containing either:
        - Dictionary of filters by frequency (receiver calibration)
        - Single filter object (sensor calibration)

        Parameters
        ----------
        cal_fn : str, pathlib.Path, or None, optional
            Path to the calibration file to read. If None, uses the previously
            set calibration file path.

        Raises
        ------
        IOError
            If the calibration file cannot be found or read.

        Notes
        -----
        The method automatically determines calibration type based on file_type:
        - "receiver calibration": Creates multiple filters per channel by frequency
        - "sensor calibration": Creates single filter per channel
        """
        if cal_fn is not None:
            self._cal_fn = Path(cal_fn)

        if not self.cal_fn.exists():
            raise IOError(f"Could not find {self.cal_fn}")

        self.obj = read_json_to_object(self.cal_fn)

        for channel in self.obj.cal_data:
            comp = channel.tag.lower()
            ch_cal_dict = {}
            for cal in channel.chan_data:
                ch_fap = FrequencyResponseTableFilter()  # type: ignore
                ch_fap.frequencies = cal.freq_Hz
                ch_fap.amplitudes = cal.magnitude
                ch_fap.phases = np.deg2rad(cal.phs_deg)

                max_freq = self.get_max_freq(ch_fap.frequencies)
                if self.obj.file_type in ["receiver calibration"]:
                    ch_fap.name = self.get_filter_lp_name(comp, max_freq)
                else:
                    ch_fap.name = self.get_filter_sensor_name(self.obj.sensor_serial)
                ch_fap.calibration_date = self.obj.timestamp_utc
                ch_cal_dict[max_freq] = ch_fap
                ch_fap.units_in = "Volt"
                ch_fap.units_out = "Volt"

            if "sensor" in self.obj.file_type:
                ch_fap.units_in = "milliVolt"
                ch_fap.units_out = "nanoTesla"
                setattr(self, comp, ch_fap)

            else:
                setattr(self, comp, ch_cal_dict)

    def get_filter(
        self, channel: str, filter_name: str | int
    ) -> FrequencyResponseTableFilter:
        """
        Get the frequency response filter for a specific channel and filter.

        Retrieves the lowpass filter for the given channel and filter specification.
        The method automatically handles both string and integer filter names.

        Parameters
        ----------
        channel : str
            Channel identifier (e.g., 'e1', 'h2', 'h3').
        filter_name : str or int
            Filter specification, typically the lowpass frequency in Hz
            (e.g., 1000, '100', 10000).

        Returns
        -------
        FrequencyResponseTableFilter
            The frequency response filter object containing the calibration data
            for the specified channel and filter.

        Raises
        ------
        AttributeError
            If the specified channel is not found in the calibration data.
        KeyError
            If the specified filter is not found for the given channel.

        Examples
        --------
        >>> cal = PhoenixCalibration("calibration.json")
        >>> filt = cal.get_filter("e1", 1000)
        >>> print(f"Filter name: {filt.name}")
        >>> print(f"Frequency points: {len(filt.frequencies)}")
        """
        try:
            filter_name = int(filter_name)
        except ValueError:
            pass

        try:
            return getattr(self, channel)[filter_name]
        except AttributeError:
            raise AttributeError(f"Could not find {channel}")
        except KeyError:
            raise KeyError(f"Could not find lowpass filter {filter_name}")
