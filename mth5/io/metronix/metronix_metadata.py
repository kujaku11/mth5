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


2026-02-24
Updating to handle old .ats files.    
Metronix Sample data of both flavors can be found in the MTHotel repository at:
MTHotel/cpp/cpp/doc/old_survey_structure/Northern_Mining/ts/Sarıçam/meas_2009-08-20_13-22-00/
MTHotel/cpp/cpp/doc/new_survey_structure/Northern_Mining/stations/Sarıçam/run_001/

The filename conventions are slightly different between the old and new formats, 

.ats example:
084_V01_C00_R001_TEx_BL_2048H.ats

.atss example:
084_ADU-07e_C000_TEx_128Hz.atss

Created on Fri Nov 22 13:23:42 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import json
import pathlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Union

import defusedxml.ElementTree as ET
import numpy as np
from loguru import logger
from mt_metadata.timeseries import AppliedFilter, Electric, Magnetic, Run
from mt_metadata.timeseries.filters import ChannelResponse, FrequencyResponseTableFilter


# =============================================================================
SUPPORTED_TS_FILE_TYPES = ["atss", "ats"]

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
        if fn.suffix in [".json", ".atss"]:
            self.system_number = fn_list[0]
            self.system_name = fn_list[1]
            self.channel_number = self._parse_channel_number(fn_list[2])
            self.component = self._parse_component(fn_list[3])
            self.sample_rate = self._parse_sample_rate(fn_list[4])
            self.file_type = self._get_file_type(fn)
        elif fn.suffix in [".ats"]:
            self.system_number = fn_list[0]
            self.system_name = fn_list[1]
            self.channel_number = self._parse_channel_number(fn_list[2])
            self.component = self._parse_component(fn_list[4])
            self.sample_rate = self._parse_sample_rate(fn_list[6])
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
        elif "h" in value.lower():
            # add to handle .ats files.
            return float(value.lower().replace("h", ""))

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
        elif value.suffix.lstrip(".") in SUPPORTED_TS_FILE_TYPES:
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

        count = 0
        for f in self.metadata.filter.split(","):
            f = f.strip()
            if not f:
                continue
            count += 1
            metadata_object.add_filter(
                AppliedFilter(name=f, applied=True, stage=count)
            )
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


class MetronixRunXML:
    """
    Parse Metronix ATS run XML files and extract mt_metadata objects.

    This class reads XML metadata files that accompany old-style Metronix .ats
    time series files. The XML contains run-level metadata including start/stop
    times, sample rate, channel configurations, and calibration data.

    Parameters
    ----------
    xml_file : pathlib.Path or str, optional
        Path to the Metronix XML file

    Attributes
    ----------
    xml_file : pathlib.Path
        Path to the XML file
    _root : xml.etree.ElementTree.Element
        Root element of parsed XML
    _run_metadata : dict
        Run-level metadata (start/stop times, sample_rate, n_chans)
    _channels : dict
        Per-channel metadata keyed by channel id
    _comments : dict
        Survey/site comments from XML
    _calibrations : dict
        Per-channel calibration data

    Examples
    --------
    >>> xml = MetronixRunXML("run_001.xml")
    >>> print(xml.sample_rate)
    128.0
    >>> ex_metadata = xml.get_channel_metadata(0)  # Get Ex channel
    >>> run_metadata = xml.get_run_metadata()
    """

    # XML paths - may vary between ADU versions
    RECORDING_KEY = "./recording"
    GLOBAL_CONFIG_KEY = "./input/ADU07Hardware/global_config"
    CHANNEL_INPUT_KEY = "./recording/input/ADU07Hardware/channel_config/channel"
    # Try multiple output paths (ProcessingTree vs ProcessingTree1, id variations)
    CHANNEL_OUTPUT_KEYS = [
        "./recording/output/ProcessingTree/output/ATSWriter/configuration/channel",
        "./recording/output/ProcessingTree1/output/ATSWriter/configuration/channel",
    ]
    ATSWRITER_CONFIG_KEY = "./recording/output/ProcessingTree/output/ATSWriter/configuration"
    COMMENTS_KEY = "./recording/output/ProcessingTree/output/ATSWriter/comments"
    CALIBRATION_KEY = "./calibration_channels/channel"

    # Channel type mapping
    ELECTRIC_CHANS = ["Ex", "Ey", "Ez"]
    MAGNETIC_CHANS = ["Hx", "Hy", "Hz"]
    DIPOLE_KEYS = {
        "Ex": (["pos_x1", "pos_x2"], "x"),
        "Ey": (["pos_y1", "pos_y2"], "y"),
        "Ez": (["pos_z1", "pos_z2"], "z"),
    }

    def __init__(self, xml_file: Union[pathlib.Path, str, None] = None) -> None:
        self.xml_file: pathlib.Path | None = None
        self._root = None
        self._run_metadata: Dict[str, Any] = {}
        self._channels: Dict[int, Dict[str, Any]] = {}
        self._channel_inputs: Dict[int, Dict[str, Any]] = {}
        self._comments: Dict[str, str] = {}
        self._calibrations: Dict[int, List[Dict[str, Any]]] = {}

        if xml_file is not None:
            self.read(xml_file)

    def __str__(self) -> str:
        """Return string representation."""
        if self.xml_file is None:
            return "MetronixRunXML: No file loaded"
        lines = [f"MetronixRunXML: {self.xml_file.name}"]
        lines.append(f"  Start: {self.start_time}")
        lines.append(f"  Stop:  {self.stop_time}")
        lines.append(f"  Sample Rate: {self.sample_rate} Hz")
        lines.append(f"  Channels: {self.n_channels}")
        for ch_id, ch_data in self._channels.items():
            lines.append(f"    [{ch_id}] {ch_data.get('channel_type', '?')}: {ch_data.get('ats_data_file', '?')}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def read(self, xml_file: Union[pathlib.Path, str]) -> None:
        """
        Parse the Metronix run XML file.

        Parameters
        ----------
        xml_file : pathlib.Path or str
            Path to the XML file

        Raises
        ------
        FileNotFoundError
            If XML file does not exist
        ValueError
            If XML structure is not recognized
        """
        self.xml_file = pathlib.Path(xml_file)
        if not self.xml_file.exists():
            raise FileNotFoundError(f"XML file {self.xml_file} does not exist")

        tree = ET.parse(str(self.xml_file))
        self._root = tree.getroot()

        self._parse_run_metadata()
        self._parse_channel_inputs()
        self._parse_channel_outputs()
        self._parse_comments()
        self._parse_calibrations()

    def _parse_run_metadata(self) -> None:
        """Parse run-level metadata (times, sample rate, etc.)."""
        rec = self._root.find(self.RECORDING_KEY)
        if rec is None:
            raise ValueError(f"Cannot find {self.RECORDING_KEY} in XML")

        # Parse times
        start_date = rec.findtext("start_date", "")
        start_time = rec.findtext("start_time", "")
        stop_date = rec.findtext("stop_date", "")
        stop_time = rec.findtext("stop_time", "")

        self._run_metadata["start_datetime"] = f"{start_date} {start_time}"
        self._run_metadata["stop_datetime"] = f"{stop_date} {stop_time}"

        # Parse global config
        glo = rec.find(self.GLOBAL_CONFIG_KEY)
        if glo is not None:
            self._run_metadata["sample_rate"] = float(glo.findtext("sample_freq", "0"))
            self._run_metadata["n_channels"] = int(glo.findtext("meas_channels", "0"))
        else:
            logger.warning("Could not find global_config in XML")

    def _parse_channel_inputs(self) -> None:
        """Parse input channel configuration (gains, chopper settings)."""
        for channel_elem in self._root.findall(self.CHANNEL_INPUT_KEY):
            ch_id = int(channel_elem.get("id", -1))
            if ch_id < 0:
                continue

            self._channel_inputs[ch_id] = {
                "gain_stage1": int(channel_elem.findtext("gain_stage1", "1")),
                "gain_stage2": int(channel_elem.findtext("gain_stage2", "1")),
                "filter_type": channel_elem.findtext("filter_type", ""),
                "hchopper": int(channel_elem.findtext("hchopper", "0")),
                "echopper": int(channel_elem.findtext("echopper", "0")),
            }

    def _parse_channel_outputs(self) -> None:
        """Parse output channel configuration from ATSWriter section."""
        channel_outputs = []

        # Try different XML paths (structure varies between ADU versions)
        for key in self.CHANNEL_OUTPUT_KEYS:
            channel_outputs = self._root.findall(key)
            if channel_outputs:
                logger.debug(f"Found channel outputs at {key}")
                break

        if not channel_outputs:
            # Try finding ATSWriter and getting channels from there
            for pt_path in ["./recording/output/ProcessingTree",
                           "./recording/output/ProcessingTree1"]:
                pt = self._root.find(pt_path)
                if pt is not None:
                    ats_writer = pt.find("output/ATSWriter")
                    if ats_writer is not None:
                        config = ats_writer.find("configuration")
                        if config is not None:
                            channel_outputs = config.findall("channel")
                            if channel_outputs:
                                break

        if not channel_outputs:
            logger.warning("Could not find channel output configuration in XML")
            return

        for channel_elem in channel_outputs:
            ch_id = int(channel_elem.get("id", -1))
            if ch_id < 0:
                continue

            ch_data = {
                "channel_type": channel_elem.findtext("channel_type", ""),
                "ats_data_file": channel_elem.findtext("ats_data_file", ""),
                "sample_rate": float(channel_elem.findtext("sample_freq", str(self.sample_rate))),
                "n_samples": int(channel_elem.findtext("num_samples", "0")),
                "ts_lsb": float(channel_elem.findtext("ts_lsb", "1.0")),
                "sensor_type": channel_elem.findtext("sensor_type", ""),
                "sensor_serial": channel_elem.findtext("sensor_sernum", ""),
            }

            # Parse dipole positions for electric channels
            chan_type = ch_data["channel_type"]
            if chan_type in self.DIPOLE_KEYS:
                pos_keys, _ = self.DIPOLE_KEYS[chan_type]
                pos1 = float(channel_elem.findtext(pos_keys[0], "0"))
                pos2 = float(channel_elem.findtext(pos_keys[1], "0"))
                ch_data["dipole_length"] = abs(pos1) + abs(pos2)
                ch_data["pos1"] = pos1
                ch_data["pos2"] = pos2
            else:
                # Magnetic channels - still parse positions if present
                ch_data["pos_x1"] = float(channel_elem.findtext("pos_x1", "0"))
                ch_data["pos_y1"] = float(channel_elem.findtext("pos_y1", "0"))
                ch_data["pos_z1"] = float(channel_elem.findtext("pos_z1", "0"))

            # Merge with input config if available
            if ch_id in self._channel_inputs:
                ch_data.update(self._channel_inputs[ch_id])

            self._channels[ch_id] = ch_data

    def _parse_comments(self) -> None:
        """Parse survey/site comments."""
        # Try multiple paths
        for path in [self.COMMENTS_KEY,
                     "./recording/output/ProcessingTree1/output/ATSWriter/comments"]:
            comments = self._root.find(path)
            if comments is not None:
                break

        if comments is None:
            return

        self._comments = {
            "site_name": comments.findtext("site_name", ""),
            "client": comments.findtext("client", ""),
            "contractor": comments.findtext("contractor", ""),
            "area": comments.findtext("area", ""),
            "survey_id": comments.findtext("survey_id", ""),
            "operator": comments.findtext("operator", ""),
            "weather": comments.findtext("weather", ""),
            "general_comments": comments.findtext("general_comments", ""),
        }

    def _parse_calibrations(self) -> None:
        """Parse per-channel calibration data."""
        for channel_elem in self._root.findall(self.CALIBRATION_KEY):
            ch_id = int(channel_elem.get("id", -1))
            if ch_id < 0:
                continue

            cal_list = []
            for cal_elem in channel_elem.findall("calibration"):
                # Check for caldata elements (sensor response)
                for caldata in cal_elem.findall("caldata"):
                    chopper = caldata.get("chopper", "unknown")
                    cal_entry = {
                        "chopper": chopper,
                        "frequency": float(caldata.findtext("c1", "0")),
                        "amplitude": float(caldata.findtext("c2", "0")),
                        "phase_deg": float(caldata.findtext("c3", "0")),
                    }
                    # Get units from attributes
                    c2_elem = caldata.find("c2")
                    if c2_elem is not None:
                        cal_entry["amplitude_unit"] = c2_elem.get("unit", "")
                    c3_elem = caldata.find("c3")
                    if c3_elem is not None:
                        cal_entry["phase_unit"] = c3_elem.get("unit", "deg")
                    cal_list.append(cal_entry)

                # Also check for calibrated_item (filter names)
                cal_item = cal_elem.find("calibrated_item")
                if cal_item is not None:
                    ci = cal_item.findtext("ci", "")
                    if ci:
                        cal_list.append({"filter_name": ci})

            if cal_list:
                self._calibrations[ch_id] = cal_list

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return self._run_metadata.get("sample_rate", 0.0)

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self._run_metadata.get("n_channels", len(self._channels))

    @property
    def start_time(self) -> str:
        """Start datetime string."""
        return self._run_metadata.get("start_datetime", "")

    @property
    def stop_time(self) -> str:
        """Stop datetime string."""
        return self._run_metadata.get("stop_datetime", "")

    @property
    def start_datetime(self) -> datetime | None:
        """Start time as datetime object."""
        try:
            return datetime.strptime(self.start_time, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return None

    @property
    def stop_datetime(self) -> datetime | None:
        """Stop time as datetime object."""
        try:
            return datetime.strptime(self.stop_time, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return None

    @property
    def site_name(self) -> str:
        """Site name from comments."""
        return self._comments.get("site_name", "")

    @property
    def survey_id(self) -> str:
        """Survey ID from comments."""
        return self._comments.get("survey_id", "")

    @property
    def channel_ids(self) -> List[int]:
        """List of channel IDs."""
        return list(self._channels.keys())

    # -------------------------------------------------------------------------
    # Channel access methods
    # -------------------------------------------------------------------------
    def get_channel_info(self, channel_id: int) -> Dict[str, Any]:
        """
        Get raw channel information dictionary.

        Parameters
        ----------
        channel_id : int
            Channel ID (0-based index)

        Returns
        -------
        dict
            Channel metadata dictionary
        """
        return self._channels.get(channel_id, {})

    def get_channel_type(self, channel_id: int) -> str:
        """
        Get channel type (electric/magnetic).

        Parameters
        ----------
        channel_id : int
            Channel ID

        Returns
        -------
        str
            'electric', 'magnetic', or 'unknown'
        """
        ch_type = self._channels.get(channel_id, {}).get("channel_type", "")
        if ch_type in self.ELECTRIC_CHANS:
            return "electric"
        elif ch_type in self.MAGNETIC_CHANS:
            return "magnetic"
        return "unknown"

    def get_ats_filename(self, channel_id: int) -> str:
        """Get the .ats filename for a channel."""
        return self._channels.get(channel_id, {}).get("ats_data_file", "")

    def get_scaling_factor(self, channel_id: int) -> float:
        """Get the LSB scaling factor for a channel."""
        return self._channels.get(channel_id, {}).get("ts_lsb", 1.0)

    def get_n_samples(self, channel_id: int) -> int:
        """Get number of samples for a channel."""
        return self._channels.get(channel_id, {}).get("n_samples", 0)

    def get_chopper_state(self, channel_id: int) -> str:
        """Get chopper state for a channel ('on' or 'off')."""
        ch_info = self._channels.get(channel_id, {})
        ch_type = self.get_channel_type(channel_id)
        if ch_type == "magnetic":
            return "on" if ch_info.get("hchopper", 0) else "off"
        elif ch_type == "electric":
            return "on" if ch_info.get("echopper", 0) else "off"
        return "unknown"

    # -------------------------------------------------------------------------
    # mt_metadata object creation
    # -------------------------------------------------------------------------
    def get_channel_metadata(self, channel_id: int) -> Union[Electric, Magnetic, None]:
        """
        Create mt_metadata Electric or Magnetic object for a channel.

        Parameters
        ----------
        channel_id : int
            Channel ID (0-based index)

        Returns
        -------
        Electric or Magnetic or None
            mt_metadata channel object, or None if channel not found

        Raises
        ------
        ValueError
            If channel type is not recognized
        """
        if channel_id not in self._channels:
            logger.warning(f"Channel {channel_id} not found in XML")
            return None

        ch_info = self._channels[channel_id]
        ch_type_str = ch_info.get("channel_type", "")
        component = ch_type_str.lower()  # Ex -> ex, Hx -> hx, etc.

        if ch_type_str in self.ELECTRIC_CHANS:
            metadata_obj = Electric(
                component=component,
                channel_number=channel_id,
                sample_rate=ch_info.get("sample_rate", self.sample_rate),
                type="electric",
            )
            # Set dipole length
            if "dipole_length" in ch_info:
                metadata_obj.dipole_length = ch_info["dipole_length"]
            # Note: Electric channels don't have a sensor attribute
            # The electrode type info goes in comments or positive/negative

        elif ch_type_str in self.MAGNETIC_CHANS:
            metadata_obj = Magnetic(
                component=component,
                channel_number=channel_id,
                sample_rate=ch_info.get("sample_rate", self.sample_rate),
                type="magnetic",
            )
            metadata_obj.sensor.manufacturer = "Metronix Geophysics"
            metadata_obj.sensor.type = ch_info.get("sensor_type", "")
            metadata_obj.sensor.id = ch_info.get("sensor_serial", "")
            metadata_obj.sensor.model = ch_info.get("sensor_type", "")

        else:
            msg = f"Unknown channel type: {ch_type_str}"
            logger.error(msg)
            raise ValueError(msg)

        # Set time period
        if self.start_datetime:
            metadata_obj.time_period.start = self.start_datetime.isoformat()
        if self.stop_datetime:
            metadata_obj.time_period.end = self.stop_datetime.isoformat()

        # Add filter information
        sensor_filter = self.get_sensor_response_filter(channel_id)
        if sensor_filter is not None:
            metadata_obj.add_filter(
                AppliedFilter(name=sensor_filter.name, applied=True, stage=1)
            )

        # Add calibration filter names
        if channel_id in self._calibrations:
            stage = 2
            for cal_entry in self._calibrations[channel_id]:
                if "filter_name" in cal_entry:
                    metadata_obj.add_filter(
                        AppliedFilter(name=cal_entry["filter_name"], applied=True, stage=stage)
                    )
                    stage += 1

        return metadata_obj

    def get_sensor_response_filter(
        self, channel_id: int, chopper: str | None = None
    ) -> FrequencyResponseTableFilter | None:
        """
        Get sensor response filter for a channel.

        Parameters
        ----------
        channel_id : int
            Channel ID
        chopper : str, optional
            Chopper state to filter by ('on' or 'off').
            If None, uses the channel's actual chopper state.

        Returns
        -------
        FrequencyResponseTableFilter or None
            Sensor response filter, or None if no calibration data
        """
        if channel_id not in self._calibrations:
            return None

        # Determine which chopper state to use
        if chopper is None:
            chopper = self.get_chopper_state(channel_id)

        # Extract frequency response data
        frequencies = []
        amplitudes = []
        phases_deg = []

        for cal_entry in self._calibrations[channel_id]:
            if "frequency" not in cal_entry:
                continue
            if cal_entry.get("chopper", chopper) != chopper:
                continue

            frequencies.append(cal_entry["frequency"])
            amplitudes.append(cal_entry["amplitude"])
            phases_deg.append(cal_entry["phase_deg"])

        if not frequencies:
            return None

        # Sort by frequency
        sort_idx = np.argsort(frequencies)
        frequencies = np.array(frequencies)[sort_idx]
        amplitudes = np.array(amplitudes)[sort_idx]
        phases_rad = np.deg2rad(np.array(phases_deg)[sort_idx])

        ch_info = self._channels.get(channel_id, {})
        sensor_type = ch_info.get("sensor_type", "unknown")

        fap = FrequencyResponseTableFilter(
            name=f"{sensor_type}_chopper_{chopper}".lower(),
            frequencies=frequencies.tolist(),
            amplitudes=amplitudes.tolist(),
            phases=phases_rad.tolist(),
        )

        return fap

    def get_run_metadata(self) -> Run:
        """
        Create mt_metadata Run object with all channels.

        Returns
        -------
        Run
            mt_metadata Run object populated with channel information
        """
        run = Run()
        run.sample_rate = self.sample_rate

        if self.start_datetime:
            run.time_period.start = self.start_datetime.isoformat()
        if self.stop_datetime:
            run.time_period.end = self.stop_datetime.isoformat()

        # Add channel metadata
        for ch_id in sorted(self._channels.keys()):
            ch_metadata = self.get_channel_metadata(ch_id)
            if ch_metadata is not None:
                ch_type = self.get_channel_type(ch_id)
                if ch_type == "electric":
                    run.add_channel(ch_metadata)
                elif ch_type == "magnetic":
                    run.add_channel(ch_metadata)

        # Set data logger info from comments
        run.data_logger.id = self._comments.get("survey_id", "")

        return run

    def get_channel_response(self, channel_id: int) -> ChannelResponse:
        """
        Get all filters needed to calibrate a channel.

        Parameters
        ----------
        channel_id : int
            Channel ID

        Returns
        -------
        ChannelResponse
            Channel response object with all calibration filters
        """
        filter_list = []
        fap = self.get_sensor_response_filter(channel_id)
        if fap is not None:
            filter_list.append(fap)
        return ChannelResponse(filters_list=filter_list)