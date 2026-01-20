# -*- coding: utf-8 -*-
"""

Created on Fri Jun 10 07:52:03 2022

:author: Jared Peacock

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger
from mt_metadata.timeseries import Station

from .helpers import read_json_to_object


if TYPE_CHECKING:
    from loguru import Logger


# =============================================================================


class PhoenixConfig:
    """
    Phoenix Geophysics configuration file reader and metadata container.

    This class reads and provides access to Phoenix MTU-5C instrument
    configuration data stored in JSON format. The configuration file contains
    recording parameters, instrument settings, and metadata used to control
    data acquisition.

    Parameters
    ----------
    fn : str, pathlib.Path, or None, optional
        Path to the Phoenix configuration file (typically config.json).
        If provided, the file will be validated for existence.
    **kwargs : Any
        Additional keyword arguments (currently unused).

    Attributes
    ----------
    fn : pathlib.Path or None
        Path to the configuration file.
    obj : Any or None
        Parsed configuration object containing all settings.
    logger : loguru.Logger
        Logger instance for debugging and error reporting.

    Examples
    --------
    >>> config = PhoenixConfig("config.json")
    >>> config.read()
    >>> station = config.station_metadata()
    >>> print(f"Station ID: {station.id}")
    """

    def __init__(self, fn: str | Path | None = None, **kwargs: Any) -> None:
        self._fn: Path | None = None
        self.obj: Any = None
        self.logger: Logger = logger
        self.fn = fn

    @property
    def fn(self) -> Path | None:
        """
        Path to the Phoenix configuration file.

        Returns
        -------
        pathlib.Path or None
            The path to the configuration file, or None if not set.
        """
        return self._fn

    @fn.setter
    def fn(self, fn: str | Path | None) -> None:
        """
        Set the configuration file path with validation.

        Parameters
        ----------
        fn : str, pathlib.Path, or None
            Path to the configuration file. If None, clears the current path.
            If provided, validates that the file exists.

        Raises
        ------
        ValueError
            If the specified file does not exist.
        """
        if fn is None:
            self._fn = None
        else:
            fn = Path(fn)
            if fn.exists():
                self._fn = Path(fn)
            else:
                raise ValueError(f"Could not find {fn}")

    def read(self, fn: str | Path | None = None) -> None:
        """
        Read and parse a Phoenix configuration file.

        Loads and parses a Phoenix MTU-5C configuration file in JSON format.
        The parsed configuration is stored in the obj attribute and provides
        access to all recording parameters and instrument settings.

        Parameters
        ----------
        fn : str, pathlib.Path, or None, optional
            Path to the configuration file to read. If None, uses the
            previously set file path from the fn property.

        Raises
        ------
        ValueError
            If no file path is provided and none was previously set.
        IOError
            If the configuration file cannot be read or parsed.

        Notes
        -----
        The configuration file should be in Phoenix JSON format containing
        recording parameters, instrument settings, and metadata.
        """
        if fn is not None:
            self.fn = fn
        self.obj = read_json_to_object(self.fn)

    def has_obj(self) -> bool:
        """
        Check if configuration data has been loaded.

        Returns
        -------
        bool
            True if configuration data is loaded, False otherwise.
        """
        return self.obj is not None

    @property
    def auto_power_enabled(self) -> Any | None:
        """
        Auto power enabled setting from configuration.

        Returns
        -------
        Any or None
            The auto power enabled setting, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.auto_power_enabled
        return None

    @property
    def config(self) -> Any | None:
        """
        Main configuration section from the configuration file.

        Returns
        -------
        Any or None
            The first configuration object containing recording parameters,
            or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.config[0]
        return None

    @property
    def empower_version(self) -> Any | None:
        """
        EMPower software version from configuration.

        Returns
        -------
        Any or None
            The EMPower software version, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.empower_version
        return None

    @property
    def mtc150_reset(self) -> Any | None:
        """
        MTC150 reset setting from configuration.

        Returns
        -------
        Any or None
            The MTC150 reset setting, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.mtc150_reset
        return None

    @property
    def network(self) -> Any | None:
        """
        Network configuration from configuration file.

        Returns
        -------
        Any or None
            The network configuration settings, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.network
        return None

    @property
    def receiver(self) -> Any | None:
        """
        Receiver configuration from configuration file.

        Returns
        -------
        Any or None
            The receiver configuration settings, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.receiver
        return None

    @property
    def schedule(self) -> Any | None:
        """
        Recording schedule from configuration file.

        Returns
        -------
        Any or None
            The recording schedule configuration, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.schedule
        return None

    @property
    def surveyTechnique(self) -> Any | None:
        """
        Survey technique setting from configuration file.

        Returns
        -------
        Any or None
            The survey technique setting, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.surveyTechnique
        return None

    @property
    def timezone(self) -> Any | None:
        """
        Timezone setting from configuration file.

        Returns
        -------
        Any or None
            The timezone setting, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.timezone
        return None

    @property
    def timezone_offset(self) -> Any | None:
        """
        Timezone offset from configuration file.

        Returns
        -------
        Any or None
            The timezone offset in hours, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.timezone_offset
        return None

    @property
    def version(self) -> Any | None:
        """
        Configuration file version from configuration file.

        Returns
        -------
        Any or None
            The configuration file version, or None if no configuration is loaded.
        """
        if self.has_obj():
            return self.obj.version
        return None

    def station_metadata(self) -> Station:
        """
        Create a Station metadata object from configuration data.

        Extracts station information from the loaded configuration and creates
        a standardized Station metadata object with basic station parameters.

        Returns
        -------
        Station
            A Station metadata object populated with configuration data including
            station ID, operator information, company name, and notes.

        Raises
        ------
        AttributeError
            If no configuration is loaded or required fields are missing.

        Notes
        -----
        The method extracts the following information from config.layout:
        - Station_Name -> station.id
        - Operator -> station.acquired_by.name
        - Company_Name -> station.acquired_by.organization
        - Notes -> station.comments

        Examples
        --------
        >>> config = PhoenixConfig("config.json")
        >>> config.read()
        >>> station = config.station_metadata()
        >>> print(f"Station: {station.id}")
        """
        s = Station()  # type: ignore

        s.id = self.config.layout.Station_Name  # type: ignore
        s.acquired_by.name = self.config.layout.Operator  # type: ignore
        s.acquired_by.organization = self.config.layout.Company_Name  # type: ignore
        s.comments = self.config.layout.Notes  # type: ignore

        return s
