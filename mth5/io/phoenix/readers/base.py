"""
Module to read and parse native Phoenix Geophysics data formats of the
MTU-5C Family.

This module implements Streamed readers for segmented-decimated continuus-decimated
and native sampling rate time series formats of the MTU-5C family.

:author: Jorge Torres-Solis

Revised 2022 by J. Peacock
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from mt_metadata.timeseries.filters import ChannelResponse, CoefficientFilter

from .calibrations import PhoenixCalibration
from .config import PhoenixConfig
from .header import Header
from .receiver_metadata import PhoenixReceiverMetadata


# =============================================================================


class TSReaderBase(Header):
    """
    Generic reader that all other readers will inherit.

    This base class provides common functionality for reading Phoenix Geophysics
    time series data files, including header parsing, file sequence management,
    and metadata handling.

    Parameters
    ----------
    path : str or Path
        Path to the time series file
    num_files : int, optional
        Number of files in the sequence, by default 1
    header_length : int, optional
        Length of file header in bytes, by default 128
    report_hw_sat : bool, optional
        Whether to report hardware saturation, by default False
    **kwargs
        Additional keyword arguments passed to parent Header class

    Attributes
    ----------
    stream : BinaryIO or None
        File stream for reading binary data
    base_path : Path
        Path to the current file
    last_seq : int
        Last sequence number in the file sequence
    rx_metadata : PhoenixReceiverMetadata or None
        Receiver metadata object
    """

    def __init__(
        self,
        path: str | Path,
        num_files: int = 1,
        header_length: int = 128,
        report_hw_sat: bool = False,
        **kwargs,
    ) -> None:
        self._seq = None
        super().__init__(
            header_length=header_length, report_hw_sat=report_hw_sat, **kwargs
        )

        self.logger = logger
        self.base_path = path
        self.last_seq = self.seq + num_files
        self.stream = None
        # Open the file passed as the first file in the sequence to stream
        self._open_file(self.base_path)
        if self._recording_id is None:
            self.recording_id = self.base_path.stem.split("_")[1]
        if self._channel_id is None:
            self.channel_id = self.base_path.stem.split("_")[2]

        self.rx_metadata = None
        self.get_receiver_metadata_object()

        if self.recmeta_file_path is not None:
            self.update_channel_map_from_recmeta()

        self._channel_metadata = None

    @property
    def base_path(self) -> Path:
        """
        Full path of the file.

        Returns
        -------
        Path
            Full path to the file
        """
        return self._base_path

    @base_path.setter
    def base_path(self, value: str | Path) -> None:
        """
        Set the full path to the file.

        Parameters
        ----------
        value : str or Path
            Full path to file

        Raises
        ------
        TypeError
            If value cannot be converted to a Path object
        """
        try:
            self._base_path = Path(value)
        except TypeError:
            raise TypeError(f"Cannot set path from {value}, bad type {type(value)}")

    @property
    def base_dir(self) -> Path:
        """
        Parent directory of the file.

        Returns
        -------
        Path
            Parent directory of the file
        """
        return self.base_path.parent

    @property
    def file_name(self) -> str:
        """
        Name of the file.

        Returns
        -------
        str
            Name of the file
        """
        return self.base_path.name

    @property
    def file_extension(self) -> str:
        """
        File extension.

        Returns
        -------
        str
            File extension including the dot
        """
        return self.base_path.suffix

    @property
    def instrument_id(self) -> str:
        """
        Instrument ID extracted from filename.

        Returns
        -------
        str
            Instrument identifier
        """
        return self.base_path.stem.split("_")[0]

    @property
    def seq(self) -> int:
        """
        Sequence number of the file.

        Returns
        -------
        int
            Sequence number extracted from filename or set value
        """
        if self._seq is None:
            return int(self.base_path.stem.split("_")[3], 16)
        return self._seq

    @seq.setter
    def seq(self, value: int) -> None:
        """
        Set the sequence number.

        Parameters
        ----------
        value : int
            Sequence number
        """
        self._seq = int(value)

    @property
    def file_size(self) -> int:
        """
        File size in bytes.

        Returns
        -------
        int
            Size of the file in bytes
        """
        return self.base_path.stat().st_size

    @property
    def max_samples(self) -> int:
        """
        Maximum number of samples in a file.

        Calculated as: (total number of bytes - header length) / frame size * n samples per frame

        Returns
        -------
        int
            Maximum number of samples in the file
        """
        return int((self.file_size - self.header_length) / 4)

    @property
    def sequence_list(self) -> list[Path]:
        """
        Get all the files in the sequence sorted by sequence number.

        Returns
        -------
        list[Path]
            List of Path objects for all files in the sequence
        """
        return sorted(list(self.base_dir.glob(f"*{self.file_extension}")))

    @property
    def config_file_path(self) -> Path | None:
        """
        Path to the config.json file.

        Returns
        -------
        Path or None
            Path to config file if it exists, None otherwise
        """
        if self.base_path is not None:
            config_fn = self.base_path.parent.parent.joinpath("config.json")
            if config_fn.exists():
                return config_fn
            else:
                self.logger.warning("Could not find config file")
        return None

    @property
    def recmeta_file_path(self) -> Path | None:
        """
        Path to the recmeta.json file.

        Returns
        -------
        Path or None
            Path to recmeta file if it exists, None otherwise
        """
        if self.base_path is not None:
            recmeta_fn = self.base_path.parent.parent.joinpath("recmeta.json")
            if recmeta_fn.exists():
                return recmeta_fn
            else:
                self.logger.warning("Could not find recmeta file")
        return None

    def _open_file(self, filename: str | Path) -> bool:
        """
        Open a given file in 'rb' mode.

        Parameters
        ----------
        filename : str or Path
            Full path to file

        Returns
        -------
        bool
            True if the file is now open, False if it is not
        """
        filename = Path(filename)

        if filename.exists():
            self.logger.debug(f"Opening {filename}")
            self.stream = open(filename, "rb")
            self.unpack_header(self.stream)
            return True
        return False

    def open_next(self) -> bool:
        """
        Open the next file in the sequence.

        Returns
        -------
        bool
            True if next file is now open, False if it is not
        """
        if self.stream is not None:
            self.stream.close()
        self.seq += 1
        self.open_file_seq(self.seq)
        if self.seq < self.last_seq:
            new_path = self.sequence_list[self.seq - 1]
            return self._open_file(new_path)
        return False

    def open_file_seq(self, file_seq_num: int | None = None) -> bool:
        """
        Open a file in the sequence given the sequence number.

        Parameters
        ----------
        file_seq_num : int, optional
            Sequence number to open, by default None

        Returns
        -------
        bool
            True if file is now open, False if it is not
        """
        if self.stream is not None:
            self.stream.close()
        if file_seq_num is not None:
            self.seq = file_seq_num
        new_path = self.sequence_list[self.seq - 1]
        return self._open_file(new_path)

    def close(self) -> None:
        """
        Close the file stream.
        """
        if self.stream is not None:
            self.stream.close()

    def get_config_object(self) -> PhoenixConfig | None:
        """
        Read a config file into an object.

        Returns
        -------
        PhoenixConfig or None
            Configuration object if config file exists, None otherwise
        """
        if self.config_file_path is not None:
            return PhoenixConfig(self.config_file_path)
        return None

    def get_receiver_metadata_object(self) -> None:
        """
        Read recmeta.json into an object and store in rx_metadata attribute.
        """
        if self.recmeta_file_path is not None and self.rx_metadata is None:
            self.rx_metadata = PhoenixReceiverMetadata(self.recmeta_file_path)

    def get_lowpass_filter_name(self) -> str | None:
        """
        Get the lowpass filter used by the receiver pre-decimation.

        Returns
        -------
        str or None
            Name of the lowpass filter if available, None otherwise
        """
        if self.recmeta_file_path is not None and self.rx_metadata is not None:
            return self.rx_metadata.obj.chconfig.chans[0].lp
        return None

    def update_channel_map_from_recmeta(self) -> None:
        """
        Update channel map from recmeta.json file.
        """
        if self.recmeta_file_path is not None and self.rx_metadata is not None:
            self.channel_map = self.rx_metadata.channel_map

    def _update_channel_metadata_from_recmeta(self) -> Any:
        """
        Get channel metadata from recmeta.json.

        Returns
        -------
        Any
            Channel metadata object updated with recmeta information
        """
        ch_metadata = self.get_channel_metadata()
        if self.recmeta_file_path is not None and self.rx_metadata is not None:
            rx_ch_metadata = self.rx_metadata.get_ch_metadata(self._channel_id)
            ch_metadata.update(rx_ch_metadata)
        ch_metadata.sample_rate = self.sample_rate
        ch_metadata.time_period.start = self.recording_start_time
        return ch_metadata

    def _update_run_metadata_from_recmeta(self) -> Any:
        """
        Update run metadata from recmeta.json.

        Returns
        -------
        Any
            Run metadata object updated with recmeta information
        """
        run_metadata = self.get_run_metadata()
        if self.recmeta_file_path is not None and self.rx_metadata is not None:
            rx_run_metadata = self.rx_metadata.run_metadata
            run_metadata.update(rx_run_metadata)
            run_metadata.add_channel(self.channel_metadata)
        run_metadata.update_time_period()
        return run_metadata

    def _update_station_metadata_from_recmeta(self) -> Any:
        """
        Update station metadata from recmeta.json.

        Returns
        -------
        Any
            Station metadata object updated with recmeta information
        """
        station_metadata = self.get_station_metadata()
        if self.recmeta_file_path is not None and self.rx_metadata is not None:
            rx_station_metadata = self.rx_metadata.station_metadata
            station_metadata.update(rx_station_metadata)
            station_metadata.add_run(self.run_metadata)
        station_metadata.update_time_period()
        return station_metadata

    @property
    def channel_metadata(self) -> Any:
        """
        Channel metadata updated from recmeta.

        Returns
        -------
        Any
            Channel metadata object
        """
        if self._channel_metadata is None:
            return self._update_channel_metadata_from_recmeta()
        return self._channel_metadata

    @property
    def run_metadata(self) -> Any:
        """
        Run metadata updated from recmeta.

        Returns
        -------
        Any
            Run metadata object
        """
        return self._update_run_metadata_from_recmeta()

    @property
    def station_metadata(self) -> Any:
        """
        Station metadata updated from recmeta.

        Returns
        -------
        Any
            Station metadata object
        """
        return self._update_station_metadata_from_recmeta()

    def get_receiver_lowpass_filter(self, rxcal_fn: str | Path) -> Any:
        """
        Get receiver lowpass filter from the rxcal.json file.

        Parameters
        ----------
        rxcal_fn : str or Path
            Path to the receiver calibration file

        Returns
        -------
        Any
            Filter object from calibration file

        Raises
        ------
        ValueError
            If the lowpass filter name cannot be found
        """
        rx_cal_obj = PhoenixCalibration(rxcal_fn)
        if rx_cal_obj._has_read():
            lp_name = self.get_lowpass_filter_name()
            if lp_name is None:
                msg = (
                    f"Could not find {lp_name} for channel "
                    f"{self.channel_metadata.comp}"
                )
                self.logger.error(msg)
                raise ValueError(msg)

            return rx_cal_obj.get_filter(self.channel_metadata.component, lp_name)
        else:
            self.logger.error("Phoenix RX Calibration is None. Check file path")
            return None

    def get_dipole_filter(self) -> CoefficientFilter | None:
        """
        Get dipole filter for electric field channels.

        Returns
        -------
        CoefficientFilter or None
            Dipole filter if channel has dipole length, None otherwise
        """
        ch_metadata = self.channel_metadata.copy()

        if hasattr(ch_metadata, "dipole_length"):
            dp_filter = CoefficientFilter()
            dp_filter.gain = ch_metadata.dipole_length / 1000
            dp_filter.units_in = "milliVolt"
            dp_filter.units_out = "milliVolt per kilometer"

            for f_name in ch_metadata.filter_names:
                if "dipole" in f_name:
                    dp_filter.name = f_name

            return dp_filter
        return None

    def get_sensor_filter(self, scal_fn: str | Path) -> Any:
        """
        Get sensor filter from calibration file.

        Parameters
        ----------
        scal_fn : str or Path
            Path to sensor calibration file

        Returns
        -------
        Any
            Sensor filter object

        Notes
        -----
        This method is not implemented yet.
        """
        return None

    def get_v_to_mv_filter(self) -> CoefficientFilter:
        """
        Create a filter to convert units from volts to millivolts.

        Returns
        -------
        CoefficientFilter
            Filter that converts volts to millivolts with gain of 1000
        """
        conversion = CoefficientFilter()
        conversion.units_out = "mV"
        conversion.units_in = "V"
        conversion.name = "v_to_mv"
        conversion.gain = 1e3

        return conversion

    def get_channel_response(
        self, rxcal_fn: str | Path | None = None, scal_fn: str | Path | None = None
    ) -> ChannelResponse:
        """
        Get the channel response filter.

        Parameters
        ----------
        rxcal_fn : str, Path or None, optional
            Path to receiver calibration file, by default None
        scal_fn : str, Path or None, optional
            Path to sensor calibration file, by default None

        Returns
        -------
        ChannelResponse
            Complete channel response filter chain
        """
        ch_metadata = self.channel_metadata.copy()

        filter_list = []
        if rxcal_fn is not None:
            rx_filter = self.get_receiver_lowpass_filter(rxcal_fn)
            if rx_filter is not None:
                filter_list.append(rx_filter)

        filter_list.append(self.get_v_to_mv_filter())

        if ch_metadata.type in ["magnetic"] and scal_fn is not None:
            sensor_filter = self.get_sensor_filter(scal_fn)
            if sensor_filter is not None:
                filter_list.append(sensor_filter)
            else:
                self.logger.warning(
                    "Could not find Phoenix coil sensor calibration filter "
                    f"for channel {ch_metadata.comp}"
                )

        if ch_metadata.type in ["electric"]:
            dipole_filter = self.get_dipole_filter()
            if dipole_filter is not None:
                filter_list.append(dipole_filter)

        return ChannelResponse(filters_list=filter_list)
