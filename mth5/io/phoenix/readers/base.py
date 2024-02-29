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
from pathlib import Path
from .header import Header
from .calibrations import PhoenixCalibration
from .config import PhoenixConfig
from .receiver_metadata import PhoenixReceiverMetadata

from mt_metadata.timeseries.filters import (
    CoefficientFilter,
    ChannelResponse,
)

from loguru import logger

# =============================================================================


class TSReaderBase(Header):
    """

    Generic reader that all other readers will inherit

    """

    def __init__(
        self,
        path,
        num_files=1,
        header_length=128,
        report_hw_sat=False,
        **kwargs,
    ):
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
    def base_path(self):
        """

        :return: full path of file
        :rtype: :class:`pathlib.Path`

        """
        return self._base_path

    @base_path.setter
    def base_path(self, value):
        """

        :param value: full path to file
        :type value: string or :class:`pathlib.Path`

        """
        try:
            self._base_path = Path(value)
        except TypeError:
            raise TypeError(
                f"Cannot set path from {value}, bad type {type(value)}"
            )

    @property
    def base_dir(self):
        """

        :return: parent directory of file
        :rtype: :class:`pathlib.Path`

        """
        return self.base_path.parent

    @property
    def file_name(self):
        """

        :return: name of the file
        :rtype: string

        """
        return self.base_path.name

    @property
    def file_extension(self):
        """

        :return: file extension
        :rtype: string

        """
        return self.base_path.suffix

    @property
    def instrument_id(self):
        """

        :return: instrument ID
        :rtype: string

        """
        return self.base_path.stem.split("_")[0]

    @property
    def seq(self):
        """

        :return: sequence number of the file
        :rtype: int

        """
        if self._seq is None:
            return int(self.base_path.stem.split("_")[3], 16)
        return self._seq

    @seq.setter
    def seq(self, value):
        """

        :param value: sequence number
        :type value: integer


        """
        self._seq = int(value)

    @property
    def file_size(self):
        """

        :return: file size in bytes
        :rtype: integer

        """
        return self.base_path.stat().st_size

    @property
    def max_samples(self):
        """
        Max number of samples in a file which is:

        (total number of bytes - header length) / frame size * n samples per frame

        :return: max number of samples in a file
        :rtype: int

        """
        return int((self.file_size - self.header_length) / 4)

    @property
    def sequence_list(self):
        """
        get all the files in the sequence sorted by sequence number
        """
        return sorted(list(self.base_dir.glob(f"*{self.file_extension}")))

    @property
    def config_file_path(self):
        if self.base_path is not None:
            config_fn = self.base_path.parent.parent.joinpath("config.json")
            if config_fn.exists():
                return config_fn
            else:
                self.logger.warning("Could not find config file")

    @property
    def recmeta_file_path(self):
        if self.base_path is not None:
            recmeta_fn = self.base_path.parent.parent.joinpath("recmeta.json")
            if recmeta_fn.exists():
                return recmeta_fn
            else:
                self.logger.warning("Could not find recmeta file")

    def _open_file(self, filename):
        """
        open a given file in 'rb' mode

        :param filename: full path to file
        :type filename: :class:`pathlib.Path`
        :return: boolean if the file is now open [True] or not [False]
        :rtype: boolean

        """
        filename = Path(filename)

        if filename.exists():
            self.logger.debug(f"Opening {filename}")
            self.stream = open(filename, "rb")
            self.unpack_header(self.stream)
            return True
        return False

    def open_next(self):
        """
        Open the next file in the sequence
        :return: [True] if next file is now open, [False] if it is not
        :rtype: boolean

        """
        if self.stream is not None:
            self.stream.close()
        self.seq += 1
        self.open_file_seq(self.seq)
        if self.seq < self.last_seq:
            new_path = self.sequence_list[self.seq - 1]
            return self._open_file(new_path)
        return False

    def open_file_seq(self, file_seq_num=None):
        """
        Open a file in the sequence given the sequence number
        :param file_seq_num: sequence number to open, defaults to None
        :type file_seq_num: integer, optional
        :return: [True] if next file is now open, [False] if it is not
        :rtype: boolean

        """
        if self.stream is not None:
            self.stream.close()
        if file_seq_num is not None:
            self.seq = file_seq_num
        new_path = self.sequence_list[self.seq - 1]
        return self._open_file(new_path)

    def close(self):
        """
        Close the file

        """
        if self.stream is not None:
            self.stream.close()

    def get_config_object(self):
        """
        Read a config file into an object.

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.config_file_path is not None:
            return PhoenixConfig(self.config_file_path)

    def get_receiver_metadata_object(self):
        """
        Read recmeta.json into an object

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.recmeta_file_path is not None and self.rx_metadata is None:
            self.rx_metadata = PhoenixReceiverMetadata(self.recmeta_file_path)

    def get_lowpass_filter_name(self):
        """
        Get the lowpass filter used by the receiver pre-decimation.

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self.recmeta_file_path is not None:
            return self.rx_metadata.obj.chconfig.chans[0].lp

    def update_channel_map_from_recmeta(self):
        if self.recmeta_file_path is not None:
            self.channel_map = self.rx_metadata.channel_map

    def _update_channel_metadata_from_recmeta(self):
        """
        Get channel metadata from recmeta.json

        :return: DESCRIPTION
        :rtype: TYPE

        """

        ch_metadata = self.get_channel_metadata()
        if self.recmeta_file_path is not None:
            rx_ch_metadata = self.rx_metadata.get_ch_metadata(self._channel_id)

            ch_metadata.update(rx_ch_metadata)
        ch_metadata.sample_rate = self.sample_rate
        ch_metadata.time_period.start = self.recording_start_time
        return ch_metadata

    def _update_run_metadata_from_recmeta(self):
        """
        Updata run metadata from recmeta.json

        :return: DESCRIPTION
        :rtype: TYPE

        """
        run_metadata = self.get_run_metadata()
        if self.recmeta_file_path is not None:
            rx_run_metadata = self.rx_metadata.run_metadata
            run_metadata.update(rx_run_metadata)
            run_metadata.add_channel(self.channel_metadata)
        run_metadata.update_time_period()
        return run_metadata

    def _update_station_metadata_from_recmeta(self):
        """
        Updata station metadata from recmeta.json

        :return: DESCRIPTION
        :rtype: TYPE

        """
        station_metadata = self.get_station_metadata()
        if self.recmeta_file_path is not None:
            rx_station_metadata = self.rx_metadata.station_metadata
            station_metadata.update(rx_station_metadata)
            station_metadata.add_run(self.run_metadata)
        station_metadata.update_time_period()
        return station_metadata

    @property
    def channel_metadata(self):
        """
        Channel metadata updated from recmeta

        :return: DESCRIPTION
        :rtype: TYPE

        """

        if self._channel_metadata is None:
            return self._update_channel_metadata_from_recmeta()
        return self._channel_metadata

    @property
    def run_metadata(self):
        """
        Run metadata updated from recmeta

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self._update_run_metadata_from_recmeta()

    @property
    def station_metadata(self):
        """
        station metadata updated from recmeta

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self._update_station_metadata_from_recmeta()

    def get_receiver_lowpass_filter(self, rxcal_fn):
        """
        get reciever lowpass filter from the rxcal.json file

        :param lp_name: DESCRIPTION
        :type lp_name: TYPE
        :param rxcal_fn: DESCRIPTION
        :type rxcal_fn: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        rx_cal_obj = PhoenixCalibration(rxcal_fn)
        if rx_cal_obj._has_read():
            lp_name = self.get_lowpass_filter_name()
            return rx_cal_obj.get_filter(
                self.channel_metadata.component, lp_name
            )
        else:
            self.logger.error(
                f"Could not find {lp_name} for channel "
                f"{self.channel_metadata().comp}"
            )

    def get_dipole_filter(self):
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """
        ch_metadata = self.channel_metadata.copy()

        if hasattr(ch_metadata, "dipole_length"):
            dp_filter = CoefficientFilter()
            dp_filter.gain = ch_metadata.dipole_length / 1000
            dp_filter.units_in = "millivolts"
            dp_filter.units_out = "millivolts per kilometer"

            for f_name in ch_metadata.filter.name:
                if "dipole" in f_name:
                    dp_filter.name = f_name

            return dp_filter

    def get_sensor_filter(self, scal_fn):
        """

        :param scal_fn: DESCRIPTION
        :type scal_fn: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        return

    def get_v_to_mv_filter(self):
        """
        the units are in volts, convert to millivolts

        """

        conversion = CoefficientFilter()
        conversion.units_out = "millivolts"
        conversion.units_in = "volts"
        conversion.name = "v_to_mv"
        conversion.gain = 1e3

        return conversion

    def get_channel_response(self, rxcal_fn=None, scal_fn=None):
        """
        Get the channel response filter

        :param rxcal_fn: DESCRIPTION, defaults to None
        :type rxcal_fn: TYPE, optional
        :param scal_fn: DESCRIPTION, defaults to None
        :type scal_fn: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        ch_metadata = self.channel_metadata.copy()

        filter_list = []
        if rxcal_fn is not None:
            filter_list.append(self.get_receiver_lowpass_filter(rxcal_fn))

        filter_list.append(self.get_v_to_mv_filter())

        if ch_metadata.type in ["magnetic"] and scal_fn is not None:
            filter_list.append(self.get_sensor_filter(scal_fn))

        if ch_metadata.type in ["electric"]:
            filter_list.append(self.get_dipole_filter())

        return ChannelResponse(filters_list=filter_list)
