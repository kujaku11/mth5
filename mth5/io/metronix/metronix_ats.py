# -*- coding: utf-8 -*-
"""
ATS (Audio Time Series) file reader for old-style Metronix data.

This module provides functionality to read and process Metronix ATS binary
time series files and their associated XML metadata files. ATS files contain
32-bit integer time series data with a 1024-byte header.

The ATS format consists of:
- .ats files: Binary time series data (int32 with 1024-byte header)
- .xml file: Run-level metadata in XML format (one per run, for all channels)

Notes
-----
Key differences from ATSS format:
- ATS uses int32 data type (not float64)
- Has 1024-byte header offset
- Requires LSB scaling from XML metadata
- XML metadata is per-run, not per-channel

Classes
-------
ATS : MetronixFileNameMetadata
    Main class for reading ATS files and converting to ChannelTS objects.

Functions
---------
read_ats : function
    Convenience function to read ATS file and return ChannelTS object.

Author
------
kkappler, jpeacock

Created
-------
April 2026
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from loguru import logger
from mt_metadata.timeseries import Run, Station, Survey
from mt_metadata.timeseries.auxiliary import Auxiliary
from mt_metadata.timeseries.electric import Electric
from mt_metadata.timeseries.filters import ChannelResponse
from mt_metadata.timeseries.magnetic import Magnetic

from mth5.io.metronix import MetronixFileNameMetadata, MetronixRunXML
from mth5.timeseries import ChannelTS


# =============================================================================
# Constants
# =============================================================================
ATS_HEADER_SIZE = 1024  # bytes
ATS_DTYPE = np.int32  # Raw data type


class ATS(MetronixFileNameMetadata):
    """
    ATS (Audio Time Series) file reader for old-style Metronix data.

    Handles reading and processing of Metronix ATS binary time series files
    and their associated XML metadata files. ATS files contain 32-bit integer
    data with a 1024-byte header that must be skipped.

    Parameters
    ----------
    fn : str or Path, optional
        Path to the ATS file. If provided, metadata will be automatically
        loaded if the corresponding XML file exists.
    run_xml : MetronixRunXML, optional
        Pre-loaded run XML metadata. If not provided, will attempt to
        auto-discover the XML file in the same directory.
    **kwargs
        Additional keyword arguments passed to parent class.

    Attributes
    ----------
    run_xml : MetronixRunXML
        Run-level metadata handler for the associated XML file.
    _channel_id : int or None
        Channel ID within the run (determined from filename or XML)

    Notes
    -----
    ATS files come in sets:
    - Multiple .ats files (one per channel): Binary time series data (int32)
    - One .xml file per run: Metadata for all channels

    Examples
    --------
    >>> ats = ATS('data/085_V01_C00_R001_TEx_BL_128H.ats')
    >>> data = ats.read_ats()  # Returns scaled float64 data
    >>> channel_ts = ats.to_channel_ts()
    """

    def __init__(
        self,
        fn: Union[str, Path, None] = None,
        run_xml: MetronixRunXML | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(fn=fn, **kwargs)

        self.run_xml = run_xml
        self._channel_id: int | None = None

        if self.fn is not None and self.run_xml is None:
            if self.has_metadata_file():
                self.run_xml = MetronixRunXML(self.metadata_fn)

        # Determine channel ID
        if self.fn is not None:
            self._determine_channel_id()

    def _determine_channel_id(self) -> None:
        """Determine channel ID from filename or XML matching."""
        # First try to use channel_number from filename parsing
        if self.channel_number is not None:
            self._channel_id = self.channel_number
            return

        # Otherwise, try to match by ATS filename in XML
        if self.run_xml is not None and self.fn is not None:
            for ch_id in self.run_xml.channel_ids:
                if self.run_xml.get_ats_filename(ch_id) == self.fn.name:
                    self._channel_id = ch_id
                    return

    @property
    def channel_id(self) -> int | None:
        """Channel ID within the run."""
        return self._channel_id

    @property
    def metadata_fn(self) -> Path | None:
        """
        Path to the metadata XML file.

        Searches for a .xml file in the same directory as the ATS file.
        Metronix ATS runs typically have one XML file per run directory.

        Returns
        -------
        Path or None
            Path to the XML metadata file, or None if not found.
        """
        if self.fn is None:
            return None

        # Look for XML files in the same directory
        xml_files = list(self.fn.parent.glob("*.xml"))
        if len(xml_files) == 1:
            return xml_files[0]
        elif len(xml_files) > 1:
            # Try to find one matching our run
            # Metronix XML names often contain the same system number
            for xml_file in xml_files:
                if self.system_number and self.system_number in xml_file.name:
                    return xml_file
            # Fall back to first one
            logger.warning(
                f"Multiple XML files found in {self.fn.parent}, using {xml_files[0].name}"
            )
            return xml_files[0]
        return None

    def has_metadata_file(self) -> bool:
        """
        Check if metadata XML file exists.

        Returns
        -------
        bool
            True if the metadata XML file exists, False otherwise.
        """
        return self.metadata_fn is not None and self.metadata_fn.exists()

    @property
    def scaling_factor(self) -> float:
        """
        Get the LSB (Least Significant Bit) scaling factor.

        This factor converts raw int32 counts to physical units (mV).

        Returns
        -------
        float
            Scaling factor from XML metadata, or 1.0 if not available.
        """
        if self.run_xml is not None and self._channel_id is not None:
            return self.run_xml.get_scaling_factor(self._channel_id)
        return 1.0

    @property
    def n_samples_from_xml(self) -> int:
        """Number of samples according to XML metadata."""
        if self.run_xml is not None and self._channel_id is not None:
            return self.run_xml.get_n_samples(self._channel_id)
        return 0

    @property
    def n_samples(self) -> int:
        """
        Estimated number of samples from file size.

        Accounts for 1024-byte header offset.

        Returns
        -------
        int
            Number of samples based on file size.
        """
        if self.fn is not None and self.fn.exists():
            data_bytes = self.fn.stat().st_size - ATS_HEADER_SIZE
            return data_bytes // np.dtype(ATS_DTYPE).itemsize
        return 0

    def read_ats(
        self,
        fn: Union[str, Path, None] = None,
        start: int = 0,
        stop: int = 0,
        apply_scaling: bool = True,
    ) -> np.ndarray:
        """
        Read binary ATS time series data.

        Reads 32-bit integer time series data from the ATS binary file,
        skipping the 1024-byte header. Optionally applies LSB scaling.

        Parameters
        ----------
        fn : str or Path, optional
            Path to ATS file. If None, uses the current file path.
        start : int, default 0
            Starting sample index (0-based).
        stop : int, default 0
            Ending sample index. If 0, reads to end of file.
        apply_scaling : bool, default True
            If True, multiplies data by LSB scaling factor from XML.

        Returns
        -------
        np.ndarray
            Time series data as 1D array of np.float32 (if scaled) or
            np.int32 (if not scaled) values.

        Raises
        ------
        ValueError
            If stop index exceeds the number of samples in the file.
        FileNotFoundError
            If the ATS file does not exist.

        Examples
        --------
        >>> ats = ATS('data/085_V01_C00_R001_TEx_BL_128H.ats')
        >>> data = ats.read_ats()  # Read entire file, scaled
        >>> raw_data = ats.read_ats(apply_scaling=False)  # Raw counts
        >>> data_slice = ats.read_ats(start=1000, stop=2000)  # Read subset
        """
        if fn is not None:
            self.fn = fn

        if not self.fn_exists:
            raise FileNotFoundError(f"ATS file not found: {self.fn}")

        if stop > 0 and stop > self.n_samples:
            raise ValueError(f"stop {stop} > n_samples {self.n_samples}")

        # Calculate byte offset (header + start samples)
        byte_offset = ATS_HEADER_SIZE + start * np.dtype(ATS_DTYPE).itemsize

        # Determine number of samples to read
        if stop == 0:
            n_read = self.n_samples - start
        else:
            n_read = stop - start

        # Read using memory mapping for efficiency
        data = np.memmap(
            self.fn,
            dtype=ATS_DTYPE,
            mode="r",
            offset=byte_offset,
            shape=(n_read,),
        ).astype(np.float32)

        if apply_scaling:
            data = data * self.scaling_factor

        return data

    @property
    def channel_metadata(self) -> Union[Electric, Magnetic, Auxiliary, None]:
        """
        Channel metadata from the XML header file.

        Returns
        -------
        Electric or Magnetic or Auxiliary or None
            Channel metadata object based on the channel type,
            or None if metadata is not available.
        """
        if self.run_xml is not None and self._channel_id is not None:
            return self.run_xml.get_channel_metadata(self._channel_id)
        return None

    @property
    def channel_response(self) -> ChannelResponse:
        """
        Channel response information from the XML header file.

        Returns
        -------
        ChannelResponse
            Channel response/calibration information.
        """
        if self.run_xml is not None and self._channel_id is not None:
            return self.run_xml.get_channel_response(self._channel_id)
        return ChannelResponse()

    @property
    def channel_type(self) -> str:
        """
        Determine channel type from component name.

        Returns
        -------
        str
            Channel type: 'electric', 'magnetic', or 'auxiliary'.
        """
        if self.fn_exists:
            if self.component.startswith("e"):
                return "electric"
            elif self.component.startswith("h"):
                return "magnetic"
            else:
                return "auxiliary"
        return "unknown"

    @property
    def run_id(self) -> str | None:
        """
        Extract run ID from file path.

        For Metronix ATS, the run is typically the parent directory
        (a meas_* folder like meas_2009-08-20_13-22-00).

        Returns
        -------
        str or None
            Run identifier.
        """
        if self.fn_exists:
            return self.fn.parent.name
        return None

    @property
    def station_id(self) -> str | None:
        """
        Extract station ID from file path.

        For Metronix ATS with old survey structure:
        .../survey/ts/station/meas_*/files.ats

        Returns
        -------
        str or None
            Station identifier.
        """
        if self.fn_exists:
            return self.fn.parent.parent.name
        return None

    @property
    def survey_id(self) -> str | None:
        """
        Extract survey ID from file path or XML comments.

        Returns
        -------
        str or None
            Survey identifier.
        """
        if self.run_xml is not None:
            survey_id = self.run_xml.survey_id
            if survey_id:
                return survey_id
        if self.fn_exists:
            # Try to extract from directory structure
            # .../survey/ts/station/meas_*/files.ats
            return self.fn.parent.parent.parent.parent.name
        return None

    @property
    def run_metadata(self) -> Run:
        """
        Generate run-level metadata.

        Returns
        -------
        Run
            Run metadata object with data logger info, sample rate,
            and channel metadata.
        """
        if self.run_xml is not None:
            run = self.run_xml.get_run_metadata()
            run.id = self.run_id
            return run

        # Fallback: minimal metadata from filename
        run = Run(id=self.run_id)
        run.data_logger.id = self.system_number
        run.data_logger.manufacturer = "Metronix Geophysics"
        run.data_logger.model = self.system_name
        run.sample_rate = self.sample_rate
        if self.channel_metadata is not None:
            run.add_channel(self.channel_metadata)
        run.update_time_period()
        return run

    @property
    def station_metadata(self) -> Station:
        """
        Generate station-level metadata.

        Returns
        -------
        Station
            Station metadata object with run metadata.
        """
        station = Station(id=self.station_id)
        if self.run_xml is not None:
            site_name = self.run_xml.site_name
            if site_name:
                station.geographic_name = site_name
        station.add_run(self.run_metadata)
        station.update_time_period()
        return station

    @property
    def survey_metadata(self) -> Survey:
        """
        Generate survey-level metadata.

        Returns
        -------
        Survey
            Survey metadata object containing station information.
        """
        survey = Survey(id=self.survey_id)
        survey.add_station(self.station_metadata)
        survey.update_time_period()
        return survey

    def to_channel_ts(self, fn: Union[str, Path, None] = None) -> ChannelTS:
        """
        Create a ChannelTS object from ATS data.

        Converts the ATS time series data and metadata into a ChannelTS
        object suitable for use with MTH5 workflows.

        Parameters
        ----------
        fn : str or Path, optional
            Path to ATS file. If None, uses current file path.

        Returns
        -------
        ChannelTS
            Time series object with data, metadata, and response information.

        Warnings
        --------
        A warning is logged if the metadata XML file is missing.

        Examples
        --------
        >>> ats = ATS('data/085_V01_C00_R001_TEx_BL_128H.ats')
        >>> channel_ts = ats.to_channel_ts()
        >>> print(channel_ts.sample_rate)
        128.0
        """
        if fn is not None:
            self.fn = fn

        if not self.has_metadata_file():
            logger.warning(
                f"Could not find Metronix metadata XML file for {self.fn.name}."
            )

        return ChannelTS(
            channel_type=self.channel_type,
            data=self.read_ats(),
            channel_metadata=self.channel_metadata,
            channel_response=self.channel_response,
            run_metadata=self.run_metadata,
            station_metadata=self.station_metadata,
            survey_metadata=self.survey_metadata,
        )


def read_ats(
    fn: Union[str, Path],
    run_xml: MetronixRunXML | None = None,
    calibration_fn: Union[str, Path, None] = None,
) -> ChannelTS:
    """
    Generic tool to read ATS file and return ChannelTS object.

    Convenience function that creates an ATS object and converts it
    to a ChannelTS in a single call.

    Parameters
    ----------
    fn : str or Path
        Path to the ATS file to read.
    run_xml : MetronixRunXML, optional
        Pre-loaded run XML metadata. If not provided, will attempt
        to auto-discover the XML file.
    calibration_fn : str or Path, optional
        Path to calibration file (currently unused).

    Returns
    -------
    ChannelTS
        Time series object with data and metadata from the ATS file.

    Examples
    --------
    >>> channel_ts = read_ats('data/085_V01_C00_R001_TEx_BL_128H.ats')
    >>> print(f"Loaded {len(channel_ts.ts)} samples")
    """
    ats_obj = ATS(fn, run_xml=run_xml)
    return ats_obj.to_channel_ts()
