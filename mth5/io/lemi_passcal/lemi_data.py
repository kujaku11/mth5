#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Routines for data prep before conversion to mseed files

Maeva Pourpoint - IRIS/PASSCAL
"""
# from __future__ import annotations

import glob
import logging
import logging.config
import numpy as np
import os
import re
import sys

from obspy import UTCDateTime
from typing import Dict, List, Optional, TYPE_CHECKING

from lemi2obspy.utils import convert_time, convert_coordinate, str2list, get_e_loc

if TYPE_CHECKING:
    from lemi2obspy.lemi_metadata import LemiMetadata

# Read logging config file
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.conf")
logging.config.fileConfig(log_file_path)
# Create logger
logger = logging.getLogger(__name__)

CHANNEL_NAMING_CONVENTION = {
    "Ex": "LQN",
    "Ey": "LQE",
    "Hx": "LFN",
    "Hy": "LFE",
    "Hz": "LFZ",
    "Ui": "LEH",
    "Te": "LKH",
    "Tf": "LKF",
    "Sn": "GNS",
    "Fq": "GST",
    "Ce": "LCE",
}
DATA_NBR_COLUMNS = 24
DATA_INDEX = {
    "Time": range(0, 6),
    "Hx": 6,
    "Hy": 7,
    "Hz": 8,
    "Te": 9,
    "Tf": 10,
    "E1": 11,
    "E2": 12,
    "E3": 13,
    "E4": 14,
    "Ui": 15,
    "Elev": 16,
    "Lat": 17,
    "Lat_Hem": 18,
    "Lon": 19,
    "Lon_Hem": 20,
    "Sn": 21,
    "Fq": 22,
    "Ce": 23,
}
DATA_TO_ARCHIVE = [
    "E1",
    "E2",
    "E3",
    "E4",
    "Hx",
    "Hy",
    "Hz",
    "Ui",
    "Te",
    "Tf",
    "Sn",
    "Fq",
    "Ce",
]
SAMPLING_RATE = 1.0  # Hz
VALID_COMPONENTS = ["E1", "E2", "E3", "E4", "Hx", "Hy", "Hz"]


class LemiData:
    def __init__(self, path2data: str) -> None:
        self.path2data = path2data
        self.data_files: List[str] = []
        self.stats: Dict = {
            "sample_rate": SAMPLING_RATE,
        }
        self.scan_path2data()

    def scan_path2data(self) -> None:
        data_files = [
            tmp
            for tmp in glob.glob(self.path2data + "*")
            if re.match(r"^\w{12}.txt$", os.path.split(tmp)[-1], re.IGNORECASE)
        ]
        if not data_files:
            logger.error(
                "No data files found under the following path - {}. "
                "Check data path!".format(self.path2data)
            )
            sys.exit(1)
        self.data_files = self.order_files(data_files)

    def parse_file(self, data_file) -> Optional[Dict]:
        data: Dict = {}
        msg = "The data file - {} - may have been corrupted. Skipping file!".format(
            data_file
        )
        with open(data_file, "r", newline="") as fin:
            for line in fin:
                columns = line.strip().split()
                if line.endswith("\r\n") and len(columns) == DATA_NBR_COLUMNS:
                    tmp = self.reformat_data(columns)
                    if tmp is None:
                        logger.warning(msg)
                        return None
                    for key, val in tmp.items():
                        if data.get(key) is None:
                            data[key] = [val]
                        else:
                            data[key].append(val)
                else:
                    logger.warning(msg)
                    return None
        return data

    def parse_all_files(self) -> None:
        self.data: Dict = {}
        for data_file in self.data_files:
            logger.info("Parsing data file - {}.".format(data_file))
            data = self.parse_file(data_file)
            if data is not None:
                for key, val in data.items():
                    if self.data.get(key) is None:
                        self.data[key] = val
                    else:
                        self.data[key] = [*self.data[key], *val]

    @staticmethod
    def order_files(files) -> List:
        return sorted(files, key=lambda x: x.split("/")[-1].split(".")[0])

    @staticmethod
    def reformat_data(columns) -> Optional[Dict]:
        time_stamp = convert_time(
            " ".join([columns[ind] for ind in DATA_INDEX["Time"]])
        )  # type: ignore
        lat = convert_coordinate(
            columns[DATA_INDEX["Lat"]], columns[DATA_INDEX["Lat_Hem"]]
        )
        lon = convert_coordinate(
            columns[DATA_INDEX["Lon"]], columns[DATA_INDEX["Lon_Hem"]]
        )
        if not all([time_stamp, lat, lon]):
            return None
        dict_ = {
            "Time_stamp": time_stamp,
            "Lat": lat,
            "Lon": lon,
            "Elev": float(columns[DATA_INDEX["Elev"]]),
            "Hx": float(columns[DATA_INDEX["Hx"]]),
            "Hy": float(columns[DATA_INDEX["Hy"]]),
            "Hz": float(columns[DATA_INDEX["Hz"]]),
            "E1": float(columns[DATA_INDEX["E1"]]),
            "E2": float(columns[DATA_INDEX["E2"]]),
            "E3": float(columns[DATA_INDEX["E3"]]),
            "E4": float(columns[DATA_INDEX["E4"]]),
            "Ui": float(columns[DATA_INDEX["Ui"]]),
            "Te": float(columns[DATA_INDEX["Te"]]),
            "Tf": float(columns[DATA_INDEX["Tf"]]),
            "Sn": float(columns[DATA_INDEX["Sn"]]),
            "Fq": float(columns[DATA_INDEX["Fq"]]),
            "Ce": float(columns[DATA_INDEX["Ce"]]),
        }
        return dict_

    @staticmethod
    def detect_gaps(time_stamp) -> List:
        diffs = [j - i for i, j in zip(time_stamp[:-1], time_stamp[1:])]
        ind_gap = [ind + 1 for ind, diff in enumerate(diffs) if diff != SAMPLING_RATE]
        if ind_gap:
            logger.warning(
                "Data gaps detected at {}.".format(
                    ", ".join([str(time_stamp[x]) for x in ind_gap])
                )
            )
        return ind_gap

    @staticmethod
    def detect_new_day(time_stamp) -> List:
        return [
            i + 1
            for i in range(len(time_stamp) - 1)
            if time_stamp[i + 1].day != time_stamp[i].day
        ]

    def create_data_array(self) -> None:
        # Check for data gaps and day start
        time_stamp = self.data["Time_stamp"]
        ind_gaps = self.detect_gaps(time_stamp)
        ind_days = self.detect_new_day(time_stamp)
        ind_traces = sorted(set([0, *ind_gaps, *ind_days, len(time_stamp)]))
        # For LEMIs, number of data gaps defines number of runs
        # Save that info in stats
        self.stats["nbr_runs"] = len(ind_gaps) + 1
        # Create structured numpy array
        npts_max = int(24 * 3600 * SAMPLING_RATE)
        dtype = [
            ("channel_number", str, 2),
            ("component", str, 2),
            ("channel_name", str, 3),
            ("location", str, 2),
            ("run_nbr", int),
            ("starttime", UTCDateTime),
            ("endtime", UTCDateTime),
            ("npts", int),
            ("samples", float, npts_max),
        ]
        self.data_np = np.zeros(
            len(DATA_TO_ARCHIVE) * (len(ind_traces) - 1), dtype=dtype
        )
        # Fill array for each time chunk and channel.
        # A time chunk is defined as a day or the time between two data gaps if
        # data recording is discontinuous.
        ind = 0
        run_nbr = 1
        location = ""
        for start, end in zip(ind_traces[:-1], ind_traces[1:]):
            npts = end - start
            if start in ind_gaps:
                run_nbr += 1
            for cha in DATA_TO_ARCHIVE:
                channel_number = cha if cha.startswith("E") else ""
                component = cha if not cha.startswith("E") else ""
                channel_name = CHANNEL_NAMING_CONVENTION.get(cha, "")
                samples = [*self.data[cha][start:end], *[0] * (npts_max - npts)]
                self.data_np[ind] = np.array(
                    [
                        (
                            channel_number,
                            component,
                            channel_name,
                            location,
                            run_nbr,
                            time_stamp[start],
                            time_stamp[end - 1],
                            npts,
                            samples,
                        )
                    ],
                    dtype=dtype,
                )
                ind += 1

    def update_stats_1(self) -> None:
        self.stats["latitude"] = np.mean(self.data["Lat"])
        self.stats["longitude"] = np.mean(self.data["Lon"])
        self.stats["elevation"] = np.mean(self.data["Elev"])
        self.stats["time_period_start"] = self.data["Time_stamp"][0]
        self.stats["time_period_end"] = self.data["Time_stamp"][-1]

    def update_stats_2(self, net: str, sta: str) -> None:
        self.stats["network"] = "XX" if not net else net.upper()
        self.stats["station"] = "TEST" if not sta else sta.upper()

    def update_array(self, data_channels: List, e_info: Dict) -> None:
        # Remove from data array channels for which data were not recorded
        no_record = [x for x in VALID_COMPONENTS if x not in data_channels]
        for channel in no_record:
            self.data_np = self.data_np[self.data_np["channel_number"] != channel]
        # Update component, channel_name and location code for electric channels
        e_loc = get_e_loc(e_info)
        for key, val in e_info.items():
            for ind in np.where(self.data_np["channel_number"] == key)[0]:
                self.data_np[ind]["component"] = val
                self.data_np[ind]["channel_name"] = CHANNEL_NAMING_CONVENTION[val]
                self.data_np[ind]["location"] = e_loc.get(key, "")

    def update_data(
        self, qc_inputs: Optional[Dict] = None, metadata: Optional[LemiMetadata] = None
    ) -> None:
        """
        Called either right after parse_data_files() in QC mode or after
        parsing metadata info in normal mode.
        """
        if qc_inputs is not None:
            net = qc_inputs["net"]
            sta = qc_inputs["sta"]
            data_channels = qc_inputs["data_channels"]
            e_info = qc_inputs["e_info"]
        else:
            net = metadata.survey.archive_network  # type: ignore
            sta = metadata.station.archive_id  # type: ignore
            data_channels = str2list(metadata.station.components_recorded)  # type: ignore
            e_info = {
                x.channel_number: x.component
                for x in metadata.electric  # type: ignore
                if x.channel_number in data_channels
            }  # type: ignore
        self.update_stats_2(net, sta)
        self.update_array(data_channels, e_info)

    def prep_data(self):
        self.parse_all_files()
        if not self.data:
            logger.error("No valid data found. Exiting!")
            sys.exit()
        else:
            self.create_data_array()
            self.update_stats_1()
