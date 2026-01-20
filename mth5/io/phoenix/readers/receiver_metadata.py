# -*- coding: utf-8 -*-
"""
Phoenix Geophysics receiver metadata parser for recmeta.json files.

Created on Tue Jun 20 15:06:08 2023

@author: jpeacock
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

from loguru import logger
from mt_metadata.common import Comment
from mt_metadata.timeseries import Electric, Magnetic, Run, Station, Survey
from mt_metadata.timeseries.filtered import AppliedFilter

from .helpers import read_json_to_object


if TYPE_CHECKING:
    pass


# =============================================================================


class PhoenixReceiverMetadata:
    """
    Container for Phoenix Geophysics recmeta.json metadata files.

    This class reads and parses receiver metadata from JSON configuration files
    used to control Phoenix Geophysics MTU-5C data recording systems. It provides
    methods to extract channel configurations, instrument settings, and convert
    them to standardized metadata objects.

    Parameters
    ----------
    fn : str, Path, or None, optional
        Path to the recmeta.json file. If provided, the file will be read
        automatically during initialization.
    **kwargs
        Additional keyword arguments (currently unused).

    Attributes
    ----------
    fn : Path or None
        Path to the metadata file.
    obj : SimpleNamespace or None
        Parsed JSON content as a SimpleNamespace object.
    logger : loguru.Logger
        Logger instance for error reporting.

    Raises
    ------
    IOError
        If the specified file does not exist.

    Examples
    --------
    >>> metadata = PhoenixReceiverMetadata("recmeta.json")
    >>> channel_map = metadata.channel_map
    >>> e1_config = metadata.e1_metadata

    Notes
    -----
    The class supports both electric and magnetic channel configurations
    with automatic mapping from Phoenix-specific parameter names to
    standardized metadata attributes.
    """

    def __init__(self, fn: str | Path | None = None, **kwargs: Any) -> None:
        self.fn = fn
        self.obj: SimpleNamespace | None = None

        self._e_map = {
            "tag": "component",
            "ty": "type",
            "ga": "gain",
            "sampleRate": "sample_rate",
            "pot_p": "contact_resistance.start",
            "pot_n": "contact_resistance.end",
        }

        self._h_map = {
            "tag": "component",
            "ty": "type",
            "ga": "gain",
            "sampleRate": "sample_rate",
            "type_name": "sensor.model",
            "type": "sensor.type",
            "serial": "sensor.id",
        }
        self.logger = logger

        if self.fn is not None:
            self.read()

    @property
    def fn(self) -> Path | None:
        """
        Path to the metadata file.

        Returns
        -------
        Path or None
            Path to the recmeta.json file, or None if not set.
        """
        return self._fn

    @fn.setter
    def fn(self, fn: str | Path | None) -> None:
        """
        Set the metadata file path.

        Parameters
        ----------
        fn : str, Path, or None
            Path to the metadata file.

        Raises
        ------
        IOError
            If the specified file does not exist.
        """
        if fn is None:
            self._fn = None
        else:
            fn = Path(fn)
            if fn.exists():
                self._fn = Path(fn)
            else:
                raise IOError(f"Could not find {fn}")

    @property
    def instrument_id(self) -> str | None:
        """
        Instrument identifier from metadata.

        Returns
        -------
        str or None
            Instrument ID if available, None otherwise.
        """
        if self.has_obj() and self.obj is not None:
            return self.obj.instid
        return None

    def read(self, fn: str | Path | None = None) -> None:
        """
        Read a recmeta.json file in Phoenix format.

        Parameters
        ----------
        fn : str, Path, or None, optional
            Path to the JSON file. If None, uses the current fn property.

        Raises
        ------
        IOError
            If no file path is specified or file doesn't exist.
        ValueError
            If the file cannot be parsed as JSON.
        """
        if fn is not None:
            self.fn = fn

        if self.fn is None:
            raise IOError("No file path specified")

        self.obj = read_json_to_object(self.fn)

    def has_obj(self) -> bool:
        """
        Check if metadata object is loaded.

        Returns
        -------
        bool
            True if metadata object exists, False otherwise.
        """
        return self.obj is not None

    @property
    def channel_map(self) -> dict[int, str]:
        """
        Channel mapping from index to component tag.

        Returns
        -------
        dict[int, str]
            Dictionary mapping channel indices to component tags (lowercase).

        Raises
        ------
        AttributeError
            If metadata object is not loaded or missing channel_map.
        """
        if self.has_obj() and self.obj is not None:
            return dict([(d.idx, d.tag.lower()) for d in self.obj.channel_map.mapping])
        return {}

    @property
    def lp_filter_base_name(self) -> str | None:
        """
        Base name for low-pass filter identifiers.

        Returns
        -------
        str or None
            Filter base name combining receiver info, or None if not available.
        """
        if self.has_obj() and self.obj is not None:
            return (
                f"{self.obj.receiver_commercial_name}_"
                f"{self.obj.receiver_model}_"
                f"{self.obj.instid}"
            ).lower()
        return None

    def get_ch_index(self, tag: str) -> int:
        """
        Get channel index from component tag.

        Parameters
        ----------
        tag : str
            Component tag (e.g., 'e1', 'h1', etc.).

        Returns
        -------
        int
            Channel index corresponding to the tag.

        Raises
        ------
        ValueError
            If the tag is not found in the channel map.
        AttributeError
            If metadata object is not loaded.
        """
        if self.has_obj() and self.obj is not None:
            for item in self.obj.channel_map.mapping:
                if item.tag.lower() == tag.lower():
                    return item.idx
            raise ValueError(f"Could not find {tag} in channel map.")
        raise AttributeError("No metadata object loaded")

    def get_ch_tag(self, index: int) -> str:
        """
        Get component tag from channel index.

        Parameters
        ----------
        index : int
            Channel index.

        Returns
        -------
        str
            Component tag corresponding to the index.

        Raises
        ------
        ValueError
            If the index is not found in the channel map.
        AttributeError
            If metadata object is not loaded.
        """
        if self.has_obj() and self.obj is not None:
            for item in self.obj.channel_map.mapping:
                if item.idx == index:
                    return item.tag
            raise ValueError(f"Could not find {index} in channel map.")
        raise AttributeError("No metadata object loaded")

    def _to_electric_metadata(self, tag: str) -> Electric:
        """
        Convert Phoenix configuration to Electric channel metadata.

        Parameters
        ----------
        tag : str
            Channel tag (e.g., 'e1', 'e2').

        Returns
        -------
        Electric
            Configured electric channel metadata.

        Raises
        ------
        AttributeError
            If metadata object is not loaded.
        ValueError
            If channel tag is not found.
        """
        c = Electric()  # type: ignore[call-arg]

        if self.has_obj() and self.obj is not None:
            ch = self.obj.chconfig.chans[self.get_ch_index(tag)]

            for p_key, m_value in self._e_map.items():
                if p_key == "ty":
                    m_value = "electric"
                try:
                    value = getattr(ch, p_key)
                    # Convert any numeric values to strings if mapping to string fields
                    if isinstance(value, (int, float)) and "id" in m_value:
                        value = str(value)
                    c.update_attribute(m_value, value)
                except AttributeError:
                    self.logger.error(
                        f"recmeta.json does not contain attribute '{p_key}' for "
                        f"channel '{ch.tag}'."
                    )
            c.channel_number = self.get_ch_index(tag)
            c.dipole_length = ch.length1 + ch.length2
            c.units = "V"
            c.time_period.start = self.obj.start
            c.time_period.end = self.obj.stop
            c.filters = [
                AppliedFilter(  # type: ignore[call-arg]
                    name=f"{self.lp_filter_base_name}_{ch.tag}_{int(ch.lp)}hz_lowpass",
                    applied=True,
                    stage=1,
                ),
                AppliedFilter(  # type: ignore[call-arg]
                    name="v_to_mv",
                    applied=True,
                    stage=2,
                ),
                AppliedFilter(  # type: ignore[call-arg]
                    name=f"dipole_{int(c.dipole_length)}m",
                    applied=True,
                    stage=3,
                ),
            ]
        return c

    def _to_magnetic_metadata(self, tag: str) -> Magnetic:
        """
        Convert Phoenix configuration to Magnetic channel metadata.

        Parameters
        ----------
        tag : str
            Channel tag (e.g., 'h1', 'h2', etc.).

        Returns
        -------
        Magnetic
            Configured magnetic channel metadata.

        Raises
        ------
        AttributeError
            If metadata object is not loaded or missing attributes.
        ValueError
            If channel tag is not found.
        """
        c = Magnetic()  # type: ignore[call-arg]

        if self.has_obj() and self.obj is not None:
            ch = self.obj.chconfig.chans[self.get_ch_index(tag)]

            c.channel_number = self.get_ch_index(tag)
            c.units = "V"
            c.time_period.start = self.obj.start
            c.time_period.end = self.obj.stop
            # Set manufacturer before processing other sensor attributes
            c.sensor.manufacturer = "Phoenix Geophysics"

            for p_key, m_value in self._h_map.items():
                if p_key == "ty":
                    m_value = "magnetic"
                try:
                    value = getattr(ch, p_key)
                    # Convert sensor.id from int to str if needed
                    if p_key == "serial" and isinstance(value, int):
                        value = str(value)
                    c.update_attribute(m_value, value)
                except AttributeError:
                    self.logger.error(
                        f"recmeta.json does not contain attribute '{p_key}' for "
                        f"channel '{ch.tag}'."
                    )

            # low pass filter of the receiver
            c.filters = [
                AppliedFilter(  # type: ignore[call-arg]
                    name=f"{self.lp_filter_base_name}_{ch.tag}_{int(ch.lp)}hz_lowpass",
                    applied=True,
                    stage=1,
                ),
                AppliedFilter(  # type: ignore[call-arg]
                    name="v_to_mv",
                    applied=True,
                    stage=2,
                ),
            ]
            # Add coil response filter using the raw serial value
            if hasattr(ch, "serial") and ch.serial is not None:
                c.filters.append(
                    AppliedFilter(
                        name=f"coil_{ch.serial}_response",
                        applied=True,
                        stage=3,
                        comments=Comment(
                            author="", time_stamp="1980-01-01T00:00:00+00:00", value=""
                        ),
                    )
                )

        return c

    # Channel metadata properties
    @property
    def e1_metadata(self) -> Electric:
        """Electric channel 1 metadata."""
        return self._to_electric_metadata("e1")

    @property
    def e2_metadata(self) -> Electric:
        """Electric channel 2 metadata."""
        return self._to_electric_metadata("e2")

    @property
    def h1_metadata(self) -> Magnetic:
        """Magnetic channel 1 metadata."""
        return self._to_magnetic_metadata("h1")

    @property
    def h2_metadata(self) -> Magnetic:
        """Magnetic channel 2 metadata."""
        return self._to_magnetic_metadata("h2")

    @property
    def h3_metadata(self) -> Magnetic:
        """Magnetic channel 3 metadata."""
        return self._to_magnetic_metadata("h3")

    @property
    def h4_metadata(self) -> Magnetic:
        """Magnetic channel 4 metadata."""
        return self._to_magnetic_metadata("h4")

    @property
    def h5_metadata(self) -> Magnetic:
        """Magnetic channel 5 metadata."""
        return self._to_magnetic_metadata("h5")

    @property
    def h6_metadata(self) -> Magnetic:
        """Magnetic channel 6 metadata."""
        return self._to_magnetic_metadata("h6")

    def get_ch_metadata(self, index: int) -> Electric | Magnetic:
        """
        Get channel metadata from index.

        Parameters
        ----------
        index : int
            Channel index.

        Returns
        -------
        Electric or Magnetic
            Channel metadata object corresponding to the index.

        Raises
        ------
        ValueError
            If index is not found in channel map.
        AttributeError
            If the corresponding metadata property doesn't exist.
        """
        tag = self.get_ch_tag(index)
        return getattr(self, f"{tag.lower()}_metadata")

    # Metadata object properties
    @property
    def run_metadata(self) -> Run:
        """
        Run metadata from receiver configuration.

        Returns
        -------
        Run
            Run metadata object with data logger and timing information.
        """
        r = Run()  # type: ignore[call-arg]
        if self.has_obj() and self.obj is not None:
            r.data_logger.type = self.obj.receiver_model
            r.data_logger.model = self.obj.receiver_commercial_name
            r.data_logger.firmware.version = self.obj.motherboard.mb_fw_ver
            r.data_logger.timing_system.drift = self.obj.timing.tm_drift
        return r

    @property
    def station_metadata(self) -> Station:
        """
        Station metadata from receiver configuration.

        Returns
        -------
        Station
            Station metadata object with location and acquisition information.
        """
        s = Station()  # type: ignore[call-arg]
        if self.has_obj() and self.obj is not None:
            s.id = self.obj.layout.Station_Name.replace(" ", "_")
            s.comments = self.obj.layout.Notes
            try:
                s.acquired_by.organization = self.obj.layout.Company_Name
            except AttributeError:
                pass

            s.acquired_by.name = self.obj.layout.Operator  # type: ignore[attr-defined]
            s.location.latitude = self.obj.timing.gps_lat
            s.location.longitude = self.obj.timing.gps_lon
            s.location.elevation = self.obj.timing.gps_alt
        return s

    @property
    def survey_metadata(self) -> Survey:
        """
        Survey metadata from receiver configuration.

        Returns
        -------
        Survey
            Survey metadata object with survey information.
        """
        s = Survey()  # type: ignore[call-arg]
        if self.has_obj() and self.obj is not None:
            s.id = self.obj.layout.Survey_Name
        return s
