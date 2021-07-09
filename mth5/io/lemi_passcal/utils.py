#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Various "utils" functions required for data and metadata formatting and
validation

Maeva Pourpoint - IRIS/PASSCAL
"""
# from __future__ import annotations

import logging
import logging.config
import os
import re

from obspy import UTCDateTime
from typing import Any, Dict, List, Optional

# Read logging config file
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.conf")
logging.config.fileConfig(log_file_path)
# Create logger
logger = logging.getLogger(__name__)

ELECTRODE_DEFAULT = "Borin STELTH 4 - Silver-Silver Chloride"
FLUXGATE_DEFAULT = "LEMI-039"
LOC = ["00", "01", "02", "03"]
MSG_E_CHA = (
    "If you did not use that channel number to record data, please update "
    "your list of recorded components at the station level!"
)


def convert_coordinate(coordinate, hemisphere):
    try:
        coordinate = float(coordinate) / 100
    except ValueError:
        logger.error(
            "Failed to convert geographic coordinate - {} - to "
            "decimal degrees!".format(coordinate)
        )
        return None
    if hemisphere not in ["N", "S", "E", "W"]:
        logger.error(
            "Unexpected hemisphere - {} - listed in data file!".format(hemisphere)
        )
        return None
    return -coordinate if hemisphere in ["S", "W"] else coordinate


def convert_time(time_stamp):
    try:
        time_stamp = UTCDateTime(time_stamp)
    except ValueError:
        logger.error("Failed to convert time stamp - {} - to UTC!".format(time_stamp))
        return None
    return time_stamp


def str2list(str_input: str) -> List[str]:
    return [x.strip() for x in str_input.split(",") if x != ""]


def check_email_formatting(email: str) -> bool:
    """
    Basic email check. Check for proper formatting.
    Will not catch typos or fake email addresses.
    """
    email_pattern = r"[\w\.\-_]+@[\w\.\-_]+"
    try:
        valid = re.match(email_pattern, email)
    except TypeError:
        logger.error("The provided email '{}' should be a string.".format(email))
        return False
    else:
        if valid is None:
            logger.error(
                "Invalid email. The provided email '{}' does not "
                "meet minimum formatting requirements: "
                "account@domain.".format(email)
            )
        return bool(valid)


def check_instrument_specs(specs: str, equipment: str, channel_info: str = "") -> bool:
    if specs in [ELECTRODE_DEFAULT, FLUXGATE_DEFAULT]:
        return True
    elif re.match(r"^Manufacturer: \w* - Model: \w* - Type: \w*$", specs):
        manufacturer, model, _ = [x.split(":")[-1].strip() for x in specs.split("-")]
        if manufacturer == "":
            logger.error(
                "Please provide {0} manufacturer{1}. Required "
                "metadata field.".format(equipment, channel_info)
            )
            return False
        if model == "":
            logger.error(
                "Please provide {0} model{1}. Required metadata "
                "field.".format(equipment, channel_info)
            )
            return False
        return True
    else:
        logger.error(
            "Please provide {0} specs for{1}. Required metadata "
            "field.".format(equipment, channel_info)
        )
        return False


def check_serial_number(serial_number: str, equipment: str) -> bool:
    if serial_number is not None:
        valid = re.match(r"^\w+$", str(serial_number))
        if valid is None:
            logger.error(
                "The serial number of the {} should be x "
                "alphanumeric character long.".format(equipment)
            )
        return bool(valid)
    else:
        logger.error(
            "Please provide a serial number for the {}! {}".format(equipment, MSG_E_CHA)
        )
        return False


def check_for_components_recorded(
    components_recorded: Optional[str], type_: str
) -> list:
    channel_type = "electric field" if type_ == "E" else "magnetic field"
    msg = (
        "No {0} data recorded. If you did record {0} data, please update "
        "your list of recorded components at the station level "
        "accordingly!"
    ).format(channel_type)
    if components_recorded is None:
        logger.warning(msg)
        return []
    else:
        components = str2list(components_recorded)  # type: ignore
        components_ = [x for x in components if x.startswith(type_)]
        if not components_:
            logger.warning(msg)
        return components_


def is_valid_uri(uri: str) -> bool:
    """From obpsy.core.inventory.util private function _is_valid_uri"""
    if ":" not in uri:
        return False
    scheme, path = uri.split(":", 1)
    if any(not x.strip() for x in (scheme, path)):
        return False
    return True


def is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, list) and not val:
        return True
    if isinstance(val, dict) and not val:
        return True
    return False


def get_e_loc(e_info: Dict) -> Dict:
    e_loc = {}
    components = set(e_info.values())
    for component in components:
        channel_nbr = [key for key, val in e_info.items() if val == component]
        if len(channel_nbr) > 1:
            e_loc.update({key: LOC[ind] for ind, key in enumerate(channel_nbr)})
    return e_loc
