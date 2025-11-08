# -*- coding: utf-8 -*-
"""
Metronix metadata parsing utilities.

This module provides classes for parsing and managing metadata from Metronix
ATSS (Audio Time Series System) files and associated JSON metadata files.

Classes
-------
MetronixFileNameMetadata
    Parse metadata from Metronix filename conventions
MetronixChannelJSON
    Read and parse Metronix JSON metadata files

Created on Fri Nov 22 13:23:42 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union

import numpy as np
from loguru import logger
from mt_metadata.timeseries import AppliedFilter, Electric, Magnetic
from mt_metadata.timeseries.filters import ChannelResponse, FrequencyResponseTableFilter


# =============================================================================


class MetronixFileNameMetadata:
    """
    Parse and manage metadata from Metronix filename conventions.

    This class extracts metadata information from Metronix ATSS filenames
    including system information, channel details, and file properties.

    Parameters
    ----------
    fn : Union[str, Path, None], optional
        Path to Metronix file, by default None
    **kwargs
        Additional keyword arguments (currently unused)

    Attributes
    ----------
    system_number : str or None
        System identification number
    system_name : str or None
        Name of the system
    channel_number : int or None
        Channel number (parsed from C## format)
    component : str or None
        Component designation (e.g., 'ex', 'ey', 'hx', 'hy', 'hz')
    sample_rate : float or None
        Sampling rate in Hz
    file_type : str or None
        Type of file ('metadata' or 'timeseries')
    """

    def __init__(self, fn: Union[str, Path, None] = None, **kwargs: Any) -> None:
        self.system_number: str | None = None
        self.system_name: str | None = None
        self.channel_number: int | None = None
        self.component: str | None = None
        self.sample_rate: float | None = None
        self.file_type: str | None = None

        self.fn = fn

    def __str__(self) -> str:
        """
        Return string representation of the metadata.

        Returns
        -------
        str
            Formatted string showing Metronix file information
        """
        if self.fn is not None:
            lines = [f"Metronix ATSS {self.file_type.upper()}:"]
            lines.append(f"\tSystem Name:    {self.system_name}")
            lines.append(f"\tSystem Number:  {self.system_number}")
            lines.append(f"\tChannel Number: {self.channel_number}")
            lines.append(f"\tComponent:      {self.component}")
            lines.append(f"\tSample Rate:    {self.sample_rate}")
            return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Return string representation for debugging.

        Returns
        -------
        str
            String representation of the object
        """
        return self.__str__()

    @property
    def fn(self) -> Path | None:
        """
        Get the file path.

        Returns
        -------
        Path or None
            File path object or None if not set
        """
        return self._fn

    @fn.setter
    def fn(self, value: Union[str, Path, None]) -> None:
        """
        Set the file path and parse metadata from filename.

        Parameters
        ----------
        value : Union[str, Path, None]
            File path to set
        """
        if value is None:
            self._fn = None
        else:
            self._fn = Path(value)
            self._parse_fn(self._fn)

    @property
    def fn_exists(self) -> bool:
        """
        Check if the file exists.

        Returns
        -------
        bool
            True if file exists, False otherwise
        """
        if self.fn is not None:
            return self.fn.exists()
        return False

    def _parse_fn(self, fn: Path | None) -> None:
        """
        Parse metadata from Metronix filename.

        Extracts system number, system name, channel number, component,
        sample rate, and file type from the filename following Metronix
        conventions.

        Parameters
        ----------
        fn : Path or None
            File path to parse
        """
        if fn is None:
            return

        fn_list = fn.stem.split("_")
        self.system_number = fn_list[0]
        self.system_name = fn_list[1]
        self.channel_number = self._parse_channel_number(fn_list[2])
        self.component = self._parse_component(fn_list[3])
        self.sample_rate = self._parse_sample_rate(fn_list[4])
        self.file_type = self._get_file_type(fn)

    def _parse_channel_number(self, value: str) -> int:
        """
        Parse channel number from filename component.

        Channel number is in format C## where ## is the channel number.

        Parameters
        ----------
        value : str
            Channel string in format 'C##'

        Returns
        -------
        int
            Channel number
        """
        return int(value.replace("C", "0"))

    def _parse_component(self, value: str) -> str:
        """
        Parse component designation from filename.

        Component is in format T{comp} where {comp} is the component name
        (e.g., 'ex', 'ey', 'hx', 'hy', 'hz').

        Parameters
        ----------
        value : str
            Component string in format 'T{comp}'

        Returns
        -------
        str
            Component name in lowercase
        """
        return value.replace("T", "").lower()

    def _parse_sample_rate(self, value: str) -> float:
        """
        Parse sample rate from filename component.

        Sample rate can be in format {sr}Hz (frequency) or {sr}s (period).
        For period format, returns 1/period to get frequency.

        Parameters
        ----------
        value : str
            Sample rate string (e.g., '100Hz' or '0.01s')

        Returns
        -------
        float
            Sample rate in Hz
        """
        if "hz" in value.lower():
            return float(value.lower().replace("hz", ""))
        elif "s" in value.lower():
            return 1.0 / float(value.lower().replace("s", ""))

    def _get_file_type(self, value: Path) -> str:
        """
        Determine file type from file extension.

        Parameters
        ----------
        value : Path
            File path object

        Returns
        -------
        str
            File type ('metadata' for .json, 'timeseries' for .atss)

        Raises
        ------
        ValueError
            If file type is not supported
        """
        if value.suffix in [".json"]:
            return "metadata"
        elif value.suffix in [".atss"]:
            return "timeseries"
        else:
            raise ValueError(f"Metronix file type {value} not supported.")

    @property
    def file_size(self) -> int:
        """
        Get file size in bytes.

        Returns
        -------
        int
            File size in bytes, 0 if file is None
        """
        if self.fn is not None:
            return self.fn.stat().st_size
        return 0

    @property
    def n_samples(self) -> float:
        """
        Get estimated number of samples in file.

        Assumes 8 bytes per sample (double precision).

        Returns
        -------
        float
            Estimated number of samples
        """
        return self.file_size / 8

    @property
    def duration(self) -> float:
        """
        Get estimated duration of the file in seconds.

        Returns
        -------
        float
            Duration in seconds
        """
        return self.n_samples / self.sample_rate


class MetronixChannelJSON(MetronixFileNameMetadata):
    """
    Read and parse Metronix JSON metadata files.

    This class extends MetronixFileNameMetadata to handle JSON metadata
    files containing channel configuration and calibration information.

    Parameters
    ----------
    fn : Union[str, Path, None], optional
        Path to Metronix JSON file, by default None
    **kwargs
        Additional keyword arguments passed to parent class

    Attributes
    ----------
    metadata : SimpleNamespace or None
        Parsed JSON metadata as a SimpleNamespace object
    """

    def __init__(self, fn: Union[str, Path, None] = None, **kwargs: Any) -> None:
        super().__init__(fn=fn, **kwargs)
        self.metadata: SimpleNamespace | None = None
        if self.fn is not None:
            self.read(self.fn)

    def _has_metadata(self) -> bool:
        """
        Check if metadata has been loaded.

        Returns
        -------
        bool
            True if metadata is loaded, False otherwise
        """
        if self.metadata is None:
            return False
        return True

    @MetronixFileNameMetadata.fn.setter
    def fn(self, value: Union[str, Path, None]) -> None:
        """
        Set the file path and read JSON metadata.

        Parameters
        ----------
        value : Union[str, Path, None]
            Path to JSON file

        Raises
        ------
        IOError
            If JSON file cannot be found
        """
        if value is None:
            self._fn = None

        else:
            value = Path(value)
            if not value.exists():
                raise IOError(f"Cannot find Metronix JSON file {value}")
            self._fn = value
            self._parse_fn(self._fn)
            self.read()

    def read(self, fn: Union[str, Path, None] = None) -> None:
        """
        Read JSON metadata from file.

        Parameters
        ----------
        fn : Union[str, Path, None], optional
            Path to JSON file, by default None (uses self.fn)

        Raises
        ------
        IOError
            If JSON file cannot be found
        """
        if fn is not None:
            self.fn = fn

        if not self.fn_exists:
            raise IOError(f"Cannot find Metronix JSON file {self.fn}")

        with open(self.fn, "r") as fid:
            self.metadata = json.load(fid, object_hook=lambda d: SimpleNamespace(**d))

    def get_channel_metadata(self) -> Union[Electric, Magnetic, None]:
        """
        Translate to mt_metadata.timeseries.Channel object.

        Creates either Electric or Magnetic metadata objects based on the
        component type and applies calibration filters.

        Returns
        -------
        Union[Electric, Magnetic, None]
            mt_metadata object based on component type, or None if no metadata

        Raises
        ------
        ValueError
            If component type is not recognized
        """
        if not self._has_metadata():
            return

        sensor_response_filter = self.get_sensor_response_filter()

        if self.component.startswith("e"):
            metadata_object = Electric(
                component=self.component,
                channel_number=self.channel_number,
                measurement_azimuth=self.metadata.angle,
                measurement_tilt=self.metadata.tilt,
                sample_rate=self.sample_rate,
                type="electric",
            )
            metadata_object.positive.latitude = self.metadata.latitude
            metadata_object.positive.longitude = self.metadata.longitude
            metadata_object.positive.elevation = self.metadata.elevation
            metadata_object.contact_resistance.start = self.metadata.resistance
        elif self.component.startswith("h"):
            metadata_object = Magnetic(
                component=self.component,
                channel_number=self.channel_number,
                measurement_azimuth=self.metadata.angle,
                measurement_tilt=self.metadata.tilt,
                sample_rate=self.sample_rate,
                type="magnetic",
            )
            metadata_object.location.latitude = self.metadata.latitude
            metadata_object.location.longitude = self.metadata.longitude
            metadata_object.location.elevation = self.metadata.elevation
            metadata_object.sensor.id = self.metadata.sensor_calibration.serial
            metadata_object.sensor.manufacturer = "Metronix Geophysics"
            metadata_object.sensor.type = "induction coil"
            metadata_object.sensor.model = self.metadata.sensor_calibration.sensor

        else:
            msg = f"Do not understand channel component {self.component}"
            logger.error(msg)
            raise ValueError(msg)

        metadata_object.time_period.start = self.metadata.datetime
        metadata_object.time_period.end = (
            metadata_object.time_period.start + self.duration
        )

        metadata_object.units = self.metadata.units

        for count, f in enumerate(self.metadata.filter.split(","), start=1):
            metadata_object.add_filter(AppliedFilter(name=f, applied=True, stage=count))
        if sensor_response_filter is not None:
            metadata_object.add_filter(
                AppliedFilter(
                    name=sensor_response_filter.name, applied=True, stage=count + 1
                )
            )
        #     metadata_object.filter.name = self.metadata.filter.split(",") + [
        #         sensor_response_filter.name
        #     ]
        # else:
        #     metadata_object.filter.name = self.metadata.filter.split(",")
        # metadata_object.filter.applied = [True] * len(metadata_object.filter.name)

        return metadata_object

    def get_sensor_response_filter(self) -> FrequencyResponseTableFilter | None:
        """
        Get the sensor response frequency-amplitude-phase filter.

        Creates a FrequencyResponseTableFilter from the sensor calibration
        data stored in the JSON metadata.

        Returns
        -------
        FrequencyResponseTableFilter or None
            Sensor response filter if calibration data exists, None otherwise
        """
        if not self._has_metadata():
            return

        fap = FrequencyResponseTableFilter(
            calibration_date=self.metadata.sensor_calibration.datetime,
            name=f"{self.metadata.sensor_calibration.sensor}_chopper_{self.metadata.sensor_calibration.chopper}".lower(),
            frequencies=self.metadata.sensor_calibration.f,
            amplitudes=self.metadata.sensor_calibration.a,
            units_out=self.metadata.units,
            units_in=self.metadata.sensor_calibration.units_amplitude.split("/")[-1],
        )

        if self.metadata.sensor_calibration.units_phase in ["degrees", "deg"]:
            fap.phases = np.deg2rad(self.metadata.sensor_calibration.p)
        else:
            fap.phases = self.metadata.sensor_calibration.p

        if len(fap.frequencies) > 0:
            return fap
        return None

    def get_channel_response(self) -> ChannelResponse:
        """
        Get all filters needed to calibrate the data.

        Returns
        -------
        ChannelResponse
            Channel response object containing all calibration filters
        """
        filter_list = []
        fap = self.get_sensor_response_filter()
        if fap is not None:
            filter_list.append(fap)
        return ChannelResponse(filters_list=filter_list)
