# -*- coding: utf-8 -*-
"""

Tools for FDSN standards

Created on Wed Sep 30 11:47:01 2020

:author: Jared Peacock

:license: MIT

"""

from loguru import logger

period_code_dict = {
    "F": {"min": 1000, "max": 5000},
    "G": {"min": 1000, "max": 5000},
    "D": {"min": 250, "max": 1000},
    "C": {"min": 250, "max": 1000},
    "E": {"min": 80, "max": 250},
    "S": {"min": 10, "max": 80},
    "H": {"min": 80, "max": 250},
    "B": {"min": 10, "max": 80},
    "M": {"min": 1, "max": 10},
    "L": {"min": 0.95, "max": 1.05},
    "V": {"min": 0.095, "max": 0.105},
    "U": {"min": 0.0095, "max": 0.0105},
    "R": {"min": 0.0001, "max": 0.001},
    "P": {"min": 0.00001, "max": 0.0001},
    "T": {"min": 0.000001, "max": 0.00001},
    "Q": {"min": 0, "max": 0.000001},
}

measurement_code_dict = {
    "tilt": "A",
    "creep": "B",
    "calibration": "C",
    "pressure": "D",
    "magnetics": "F",
    "gravity": "G",
    "humidity": "I",
    "temperature": "K",
    "water_current": "O",
    "electric": "Q",
    "rain_fall": "R",
    "linear_strain": "S",
    "tide": "T",
    "wind": "W",
}

measurement_code_dict_reverse = dict([(v, k) for k, v in measurement_code_dict.items()])
# Add Y as fallback for unknown/auxiliary measurements
measurement_code_dict_reverse["Y"] = "auxiliary"

orientation_code_dict = {
    "N": {"min": 0, "max": 15},
    "E": {"min": 75, "max": 90},
    "Z": {"min": 0, "max": 15},
    "1": {"min": 15, "max": 45},
    "2": {"min": 45, "max": 75},
    "3": {"min": 15, "max": 75},
}

mt_code_dict = {"magnetics": "h", "electric": "e"}


def get_location_code(channel_obj):
    """
    Get the location code given the components and channel number

    :param channel_obj: Channel object
    :type channel_obj: :class:`~mt_metadata.timeseries.Channel`
    :return: 2 character location code
    :rtype: string

    """

    location_code = "{0}{1}".format(
        channel_obj.component[0].upper(),
        channel_obj.channel_number % 10,
    )

    return location_code


def get_period_code(sample_rate):
    """
    Get the SEED sampling rate code given a sample rate

    :param sample_rate: sample rate in samples per second
    :type sample_rate: float
    :return: single character SEED sampling code
    :rtype: string

    """
    period_code = "A"
    for key, v_dict in sorted(period_code_dict.items()):
        if (sample_rate >= v_dict["min"]) and (sample_rate <= v_dict["max"]):
            period_code = key
            break
    return period_code


def get_measurement_code(measurement):
    """
    get SEED sensor code given the measurement type

    :param measurement: measurement type, e.g.
        * temperature
        * electric
        * magnetic
    :type measurement: string
    :return: single character SEED sensor code, if the measurement type has
             not been defined yet Y is returned.
    :rtype: string

    """
    sensor_code = "Y"
    for key, code in measurement_code_dict.items():
        if measurement.lower() in key:
            sensor_code = code
    return sensor_code


def get_orientation_code(azimuth, orientation="horizontal"):
    """
    Get orientation code given angle and orientation.  This is a general
    code and the true azimuth is stored in channel

    :param azimuth: angel assuming 0 is north, 90 is east, 0 is vertical down
    :type azimuth: float
    :return: single character SEED orientation code
    :rtype: string

    """
    orientation_code = "1"
    horizontal_keys = ["N", "E", "1", "2"]
    vertical_keys = ["Z", "3"]

    azimuth = abs(azimuth) % 91
    if orientation == "horizontal":
        test_keys = horizontal_keys
    elif orientation == "vertical":
        test_keys = vertical_keys
    else:
        raise ValueError(
            f"{orientation} not supported must be [ 'horizontal' | 'vertical' ]"
        )
    for key in test_keys:
        angle_min = orientation_code_dict[key]["min"]
        angle_max = orientation_code_dict[key]["max"]
        if (azimuth <= angle_max) and (azimuth >= angle_min):
            orientation_code = key
            break
    return orientation_code


def make_channel_code(channel_obj):
    """
    Make the 3 character SEED channel code

    :param channel_obj: Channel metadata
    :type channel_obj: :class:`~mt_metadata.timeseries.Channel`
    :return: 3 character channel code
    :type: string

    """

    period_code = get_period_code(channel_obj.sample_rate)
    # Try to get measurement code from component first, then fallback to type
    sensor_code = get_measurement_code(channel_obj.component)
    if sensor_code == "Y":  # If component didn't match, try type
        sensor_code = get_measurement_code(channel_obj.type)
    if "z" in channel_obj.component.lower():
        orientation_code = get_orientation_code(
            channel_obj.measurement_tilt, orientation="vertical"
        )
    else:
        orientation_code = get_orientation_code(channel_obj.measurement_azimuth)
    channel_code = "{0}{1}{2}".format(period_code, sensor_code, orientation_code)

    return channel_code


def read_channel_code(channel_code):
    """
    read FDSN channel code

    :param channel_code: DESCRIPTION
    :type channel_code: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    if len(channel_code) != 3:
        msg = "Input FDSN channel code is not proper format, should be 3 letters"
        logger.error(msg)
        raise ValueError(msg)
    try:
        period_range = period_code_dict[channel_code[0].upper()]
    except KeyError:
        msg = (
            f"Could not find period range for {channel_code[0]}. ",
            "Setting to 1",
        )
        period_range = {"min": 1, "max": 1}
    try:
        component = measurement_code_dict_reverse[channel_code[1].upper()]
    except KeyError:
        msg = f"Could not find component for {channel_code[1]}"
        logger.error(msg)
        raise ValueError(msg)
    vertical = False
    try:
        orientation = orientation_code_dict[channel_code[2].upper()]
        if channel_code[2].upper() in ["3", "Z"]:
            vertical = True
    except KeyError:
        msg = (
            f"Could not find orientation for {channel_code[2]}. ",
            "Setting to 0.",
        )
        logger.error(msg)
        raise ValueError(msg)
    return {
        "period": period_range,
        "component": component,
        "orientation": orientation,
        "vertical": vertical,
    }


def make_mt_channel(code_dict, angle_tol=15):
    """

    :param code_dict: DESCRIPTION
    :type code_dict: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    try:
        mt_comp = mt_code_dict[code_dict["component"]]
    except KeyError:
        mt_comp = code_dict["component"]
    if not code_dict["vertical"]:
        if (
            code_dict["orientation"]["min"] >= 0
            and code_dict["orientation"]["max"] <= angle_tol
        ):
            mt_dir = "x"
        elif (
            code_dict["orientation"]["min"] >= angle_tol
            and code_dict["orientation"]["max"] <= 45
        ):
            mt_dir = "1"
        if (
            code_dict["orientation"]["min"] >= (90 - angle_tol)
            and code_dict["orientation"]["max"] <= 90
        ):
            mt_dir = "y"
        elif code_dict["orientation"]["min"] >= 45 and code_dict["orientation"][
            "max"
        ] <= (90 - angle_tol):
            mt_dir = "2"
    else:
        if (
            code_dict["orientation"]["min"] >= 0
            and code_dict["orientation"]["max"] <= angle_tol
        ):
            mt_dir = "z"
        elif (
            code_dict["orientation"]["min"] >= angle_tol
            and code_dict["orientation"]["max"] <= 90
        ):
            mt_dir = "3"
    mt_code = f"{mt_comp}{mt_dir}"

    return mt_code
