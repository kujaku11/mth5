# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:36:55 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from collections import OrderedDict

from loguru import logger

from mth5.mth5 import MTH5
from mth5.timeseries import RunTS


# =============================================================================


class ZENC:
    """
    Deal with .zenc files, which are apparently used to process data in EMTF.
    It was specifically built for processing ZEN data in EMTF, but should
    work regardless of data logger.

    The format is a header and then n_channels x n_samples of float32 values

    This class will read/write .zenc files.

    You need to input the path to an existing or new MTH5 file and a
    channel map to read/write.

    The `channel_map` needs to be in the form

    .. code-block::
        channel_map = {
            "channel_1_name":
                {"survey": survey_name,
                 "station": station_name,
                 "run": run_name,
                 "channel": channel_name,
                 "channel_number": channel_number},
            "channel_2_name":
                {"survey": survey_name,
                 "station": station_name,
                 "run": run_name,
                 "channel": channel_name,
                 "channel_number": channel_number},
            ...
                }


    """

    def __init__(self, channel_map):
        self.logger = logger
        self._channel_map_keys = [
            "survey",
            "station",
            "run",
            "channel",
            "channel_number",
        ]
        self._expected_channel_order = ["hx", "hy", "hz", "ex", "ey"]
        self.channel_map = channel_map

    @property
    def channel_map(self):
        return self._channel_map

    @channel_map.setter
    def channel_map(self, value):
        """
        need to make sure channel map is in the correct format

        :param value: dictionary of channels to use
        :type value: dict

        """

        if not isinstance(value, dict):
            raise ValueError(
                f"Input channel_map must be a dictionary not type{type(value)}"
            )

        for key, kdict in value.items():
            if not isinstance(kdict, dict):
                raise ValueError(
                    f"Input channel must be a dictionary not type{type(value)}"
                )
            if sorted(kdict.keys()) != sorted(self._channel_map_keys):
                raise KeyError(
                    f"Keys of channel dictionary must be {self._channel_map_keys} "
                    f"not {kdict.keys()}."
                )
        self._channel_map = self._sort_channel_map(value)

    def _sort_channel_map(self, channel_map):
        """
        sort by channel number

        :param channel_map: DESCRIPTION
        :type channel_map: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        sorted_channel_map = OrderedDict()

        for ch in self._expected_channel_order:
            try:
                sorted_channel_map[ch] = channel_map[ch]
            except KeyError:
                self.logger.info(f"Could not find {ch} in channel_map, skipping")

        return sorted_channel_map

    def to_zenc(self, mth5_file, channel_map=None):
        """
        write out a .zenc file

        :param mth5_file: DESCRIPTION
        :type mth5_file: TYPE
        :param channel_map: DESCRIPTION, defaults to None
        :type channel_map: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if channel_map is not None:
            self.channel_map = channel_map

        with MTH5() as m:
            m.open_mth5(mth5_file, mode="r")
            ch_list = []
            ch_metadata_list = []
            for key, ch_dict in self.channel_map.items():
                ch = m.get_channel(
                    ch_dict["station"],
                    ch_dict["run"],
                    ch_dict["channel"],
                    survey=ch_dict["survey"],
                ).to_channel_ts()
                ch_list.append(ch)
                ch_metadata_list.append(self._get_ch_metadata(ch))

            run = RunTS(ch_list)

        # write out file
        # write metadata

        with open(mth5_file, "w") as fid:
            lines = self._write_metadata(run)
            fid.write("\n".join(lines))

            for ii in range(len(run.time)):
                for comp in self._expected_channel_order:
                    run.dataset[comp].data[ii]

        # write data as (hx, hy, hz, ex, ey, ...)

    def _write_metadata(self, run_ts):
        """
        write metadata for the zenc file.

        of the form

        4096
        version: 1.0
        boxNumber: 74
        samplingFrequency: 4096
        timeDataStart: 2021-07-23 08:00:14
        timeDataEnd: 2021-07-23 08:14:58
        latitude: 58.22444
        longitude: -155.66579
        altitude: 251.30000
        rx_stn: 1
        TxFreq: 0
        TxDuty: inf
        numChans: 5
        channel: 1
        component: Hx
        length:
        sensorID: 4044
        azimuth: 0
        xyz1: 0:0:0
        xyz2: 0:0:0
        units: V
        countconversion: 9.5367431640625e-10
        next channel

        :param run_ts: DESCRIPTION
        :type run_ts: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        lines = [run_ts.sample_rate]
        for key, value in self.get_run_metadata(run_ts):
            lines.append(f"{key}: {value}")

        for ch in run_ts.channels:
            for key, value in self.get_ch_metadata(run_ts[ch]):
                lines.append(f"{key}: {value}")

        return lines

    def get_run_metadata(self, run_ts):
        """
        get run metadata from RunTS object

        4096
        version: 1.0
        boxNumber: 74
        samplingFrequency: 4096
        timeDataStart: 2021-07-23 08:00:14
        timeDataEnd: 2021-07-23 08:14:58
        latitude: 58.22444
        longitude: -155.66579
        altitude: 251.30000
        rx_stn: 1
        TxFreq: 0
        TxDuty: inf
        numChans: 5

        :param run_ts: DESCRIPTION
        :type run_ts: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        run_dict = OrderedDict()
        run_dict["version"] = 1.0
        run_dict["boxNumber"] = run_ts.run_metadata.data_logger.id
        run_dict["samplingFrequency"] = run_ts.sample_rate
        run_dict["timeDataStart"] = run_ts.start
        run_dict["timeDataEnd"] = run_ts.end
        run_dict["latitude"] = run_ts.station_metadata.location.latitude
        run_dict["longitude"] = run_ts.station_metadata.location.longitude
        run_dict["altitude"] = run_ts.station_metadata.location.elevation
        run_dict["rx_stn"] = run_ts.station_metadata.id
        run_dict["TxFreq"] = 0
        run_dict["TxDuty"] = "inf"
        run_dict["numChans"] = len(run_ts.channels)

        return run_dict

    def _get_ch_metadata(self, ch):
        """
        get channel metadata from ChannelTS

        channel: 1
        component: Hx
        length:
        sensorID: 4044
        azimuth: 0
        xyz1: 0:0:0
        xyz2: 0:0:0
        units: V
        countconversion: 9.5367431640625e-10

        :param ch: DESCRIPTION
        :type ch: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        ch_dict = OrderedDict()
        ch_dict["channel"] = ch.channel_metadata.channel_number
        ch_dict["compnent"] = ch.channel_metadata.component
        if hasattr(ch.channel_metadata, "dipole_length"):
            ch_dict["length"] = ch.channel_metadata.dipole_length
        else:
            ch_dict["length"] = None

        if ch.channel_metadata.type.lower() in ["magnetic"]:
            ch_dict["sensorID"] = ch.channel_metadata.sensor.id
        else:
            ch_dict["sensorID"] = None
        ch_dict["azimuth"] = ch.channel_metadata.measurement_azimuth
        ch_dict["xyz1"] = "0:0:0"
        ch_dict["xyz2"] = "0:0:0"
        ch_dict["units"] = ch.channel_metadata.units
        ch_dict["countconversion"] = 9.5367431640625e-10

        return ch_dict
