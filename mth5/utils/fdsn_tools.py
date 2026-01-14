# -*- coding: utf-8 -*-
"""
FDSN standards tools.

Tools for working with FDSN (Incorporated Research Institutions for Seismology)
standards including SEED channel codes, period/measurement/orientation codes,
and conversions between SEED and MT (magnetotelluric) channel formats.

Notes
-----
Created on Wed Sep 30 11:47:01 2020

Author: Jared Peacock

License: MIT

References
----------
FDSN Channel Codes: https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf
"""

from __future__ import annotations

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


def get_location_code(channel_obj: object) -> str:
    """
    Generate FDSN location code from channel metadata.

    Creates a 2-character location code from the channel's component
    and channel number.

    Parameters
    ----------
    channel_obj : object
        Channel metadata object with `component` and `channel_number` attributes.
        Expected to be of type `~mt_metadata.timeseries.Channel`.

    Returns
    -------
    str
        2-character location code formatted as first letter of component
        and last digit of channel number (e.g., 'E1', 'H0').

    Examples
    --------
    Generate location code for an electric component on channel 5::

        >>> class MockChannel:
        ...     component = 'ex'
        ...     channel_number = 5
        >>> get_location_code(MockChannel())
        'E5'

    Magnetic component on channel 12 (wraps to 2)::

        >>> class MockChannel:
        ...     component = 'hx'
        ...     channel_number = 12
        >>> get_location_code(MockChannel())
        'H2'
    """
    # Type narrowing with duck typing for Mock compatibility
    assert hasattr(channel_obj, "component") and hasattr(channel_obj, "channel_number")

    location_code = "{0}{1}".format(
        channel_obj.component[0].upper(),  # type: ignore
        channel_obj.channel_number % 10,  # type: ignore
    )

    return location_code


def get_period_code(sample_rate: float) -> str:
    """
    Get SEED sampling rate code from sample rate.

    Determines the appropriate FDSN/SEED period code based on the sample
    rate in samples per second. Codes range from 'Q' (highest frequency,
    period < 1 μs) to 'F' (lowest frequency, period 1000-5000 s).

    Parameters
    ----------
    sample_rate : float
        Sample rate in samples per second.

    Returns
    -------
    str
        Single character SEED sampling rate code. Defaults to 'A' if no
        code matches the sample rate.

    Notes
    -----
    Code mapping (frequency/period ranges):
    - 'F', 'G': 1-5 kHz
    - 'D', 'C': 250-1000 Hz
    - 'E', 'H': 80-250 Hz
    - 'S', 'B': 10-80 Hz
    - 'M': 1-10 Hz
    - 'L': 0.95-1.05 Hz
    - 'V': 0.095-0.105 Hz
    - 'U': 0.0095-0.0105 Hz
    - 'R': 0.0001-0.001 Hz
    - 'P': 0.00001-0.0001 Hz
    - 'T': 0.000001-0.00001 Hz
    - 'Q': < 0.000001 Hz

    Examples
    --------
    Get code for 100 Hz sample rate::

        >>> get_period_code(100.0)
        'B'

    Get code for 1000 Hz::

        >>> get_period_code(1000.0)
        'D'

    Get code for 10 Hz (default 'A')::

        >>> get_period_code(10.0)
        'M'
    """
    period_code = "A"
    for key, v_dict in sorted(period_code_dict.items()):
        if (sample_rate >= v_dict["min"]) and (sample_rate <= v_dict["max"]):
            period_code = key
            break
    return period_code


def get_measurement_code(measurement: str) -> str:
    """
    Get SEED sensor code from measurement type.

    Maps measurement types to single-character SEED sensor codes.
    Performs case-insensitive substring matching.

    Parameters
    ----------
    measurement : str
        Measurement type (e.g., 'electric', 'magnetics', 'temperature',
        'tilt', 'pressure', 'humidity', 'gravity', 'calibration',
        'rain_fall', 'water_current', 'wind', 'linear_strain', 'tide',
        'creep').

    Returns
    -------
    str
        Single character SEED sensor code. Returns 'Y' if measurement
        type not found in mapping dictionary.

    Notes
    -----
    Measurement to code mapping:
    - 'tilt' → 'A'
    - 'creep' → 'B'
    - 'calibration' → 'C'
    - 'pressure' → 'D'
    - 'magnetics' → 'F'
    - 'gravity' → 'G'
    - 'humidity' → 'I'
    - 'temperature' → 'K'
    - 'water_current' → 'O'
    - 'electric' → 'Q'
    - 'rain_fall' → 'R'
    - 'linear_strain' → 'S'
    - 'tide' → 'T'
    - 'wind' → 'W'
    - unknown/auxiliary → 'Y'

    Examples
    --------
    Get code for electric measurement::

        >>> get_measurement_code('electric')
        'Q'

    Get code for magnetic field::

        >>> get_measurement_code('magnetics')
        'F'

    Unknown measurement returns 'Y'::

        >>> get_measurement_code('unknown')
        'Y'
    """
    sensor_code = "Y"
    for key, code in measurement_code_dict.items():
        if measurement.lower() in key:
            sensor_code = code
    return sensor_code


def get_orientation_code(azimuth: float, orientation: str = "horizontal") -> str:
    """
    Get SEED orientation code from azimuth and orientation type.

    Maps azimuth angle to SEED orientation code based on whether the
    sensor is oriented horizontally or vertically.

    Parameters
    ----------
    azimuth : float
        Azimuth angle in degrees where 0 is north, 90 is east,
        180 is south, 270 is west. For vertical orientation,
        0 = vertical down.
    orientation : {'horizontal', 'vertical'}, default 'horizontal'
        Type of sensor orientation.

    Returns
    -------
    str
        Single character SEED orientation code.

    Raises
    ------
    ValueError
        If `orientation` is not 'horizontal' or 'vertical'.

    Notes
    -----
    Horizontal orientation codes (azimuths):
    - 'N': 0-15° (North)
    - 'E': 75-90° (East)
    - '1': 15-45° (NE quadrant)
    - '2': 45-75° (SE quadrant)

    Vertical orientation codes:
    - 'Z': 0-15° (Primary vertical)
    - '3': 15-75° (Alternate vertical)

    Examples
    --------
    Get code for northerly azimuth::

        >>> get_orientation_code(10.0, orientation='horizontal')
        'N'

    Get code for easterly azimuth::

        >>> get_orientation_code(85.0, orientation='horizontal')
        'E'

    Get code for vertical sensor::

        >>> get_orientation_code(0.0, orientation='vertical')
        'Z'
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


def make_channel_code(channel_obj: object) -> str:
    """
    Generate 3-character SEED channel code from channel metadata.

    Combines period code, measurement code, and orientation code into
    a standard 3-character FDSN channel code.

    Parameters
    ----------
    channel_obj : object
        Channel metadata object with attributes: `sample_rate`, `component`,
        `type`, `measurement_azimuth`, `measurement_tilt`.
        Expected to be of type `~mt_metadata.timeseries.Channel`.

    Returns
    -------
    str
        3-character SEED channel code (e.g., 'BHZ', 'HHE').

    Notes
    -----
    The channel code format is: [Period Code][Measurement Code][Orientation Code]

    - Period code: based on sample_rate
    - Measurement code: derived from component, with fallback to type
    - Orientation code: depends on whether component is vertical ('z')

    Examples
    --------
    Create channel code for horizontal electric component::

        >>> class MockChannel:
        ...     sample_rate = 100.0
        ...     component = 'ex'
        ...     type = 'electric'
        ...     measurement_azimuth = 0.0
        ...     measurement_tilt = 0.0
        >>> make_channel_code(MockChannel())
        'BQN'

    Create channel code for vertical magnetic component::

        >>> class MockChannel:
        ...     sample_rate = 100.0
        ...     component = 'hz'
        ...     type = 'magnetic'
        ...     measurement_azimuth = 0.0
        ...     measurement_tilt = 0.0
        >>> make_channel_code(MockChannel())
        'BFZ'
    """
    # Type narrowing with duck typing for Mock compatibility
    assert (
        hasattr(channel_obj, "sample_rate")
        and hasattr(channel_obj, "component")
        and hasattr(channel_obj, "type")
        and hasattr(channel_obj, "measurement_azimuth")
        and hasattr(channel_obj, "measurement_tilt")
    )

    period_code = get_period_code(channel_obj.sample_rate)  # type: ignore
    # Try to get measurement code from component first, then fallback to type
    sensor_code = get_measurement_code(channel_obj.component)  # type: ignore
    if sensor_code == "Y":  # If component didn't match, try type
        sensor_code = get_measurement_code(channel_obj.type)  # type: ignore
    if "z" in channel_obj.component.lower():  # type: ignore
        orientation_code = get_orientation_code(
            channel_obj.measurement_tilt, orientation="vertical"  # type: ignore
        )
    else:
        orientation_code = get_orientation_code(channel_obj.measurement_azimuth)  # type: ignore
    channel_code = "{0}{1}{2}".format(period_code, sensor_code, orientation_code)

    return channel_code


def read_channel_code(channel_code: str) -> dict[str, dict[str, int] | str | bool]:
    """
    Parse FDSN channel code into components.

    Decodes a 3-character SEED channel code into its constituent parts:
    period range, component type, orientation range, and vertical flag.

    Parameters
    ----------
    channel_code : str
        3-character FDSN channel code (e.g., 'BHZ', 'HHE').

    Returns
    -------
    dict
        Dictionary with keys:
        - 'period' (dict): Period range with 'min' and 'max' keys (Hz).
        - 'component' (str): Component type (e.g., 'electric', 'magnetics').
        - 'orientation' (dict): Angle range with 'min' and 'max' keys (degrees).
        - 'vertical' (bool): True if component is vertical, False otherwise.

    Raises
    ------
    ValueError
        If channel code is not 3 characters, contains invalid period code,
        or contains invalid orientation code.

    Notes
    -----
    Vertical components are identified by orientation codes 'Z' or '3'.

    Examples
    --------
    Decode a horizontal channel code::

        >>> result = read_channel_code('BHE')
        >>> result['component']
        'magnetics'
        >>> result['vertical']
        False

    Decode a vertical channel code::

        >>> result = read_channel_code('BHZ')
        >>> result['vertical']
        True
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


def make_mt_channel(
    code_dict: dict[str, dict[str, int] | str | bool], angle_tol: int = 15
) -> str:
    """
    Convert FDSN code dictionary to magnetotelluric (MT) channel code.

    Maps FDSN codes to MT channel naming convention (e.g., 'ex', 'hy', 'hz').

    Parameters
    ----------
    code_dict : dict
        Dictionary with keys:
        - 'component' (str): Measurement type (e.g., 'magnetics', 'electric').
        - 'vertical' (bool): True for vertical, False for horizontal.
        - 'orientation' (dict): Orientation range with 'min' and 'max'.
    angle_tol : int, default 15
        Angle tolerance in degrees for determining cardinal directions.

    Returns
    -------
    str
        2-character MT channel code (e.g., 'ex', 'hy', 'hz').
        Format: [component code][direction code]
        - Component: 'e' (electric) or 'h' (magnetic)
        - Direction: 'x', 'y', 'z' for cardinal or '1', '2', '3' for intermediate

    Notes
    -----
    Direction mapping for horizontal channels (0-90°):
    - 'x': North direction (0-15°)
    - '1': NE quadrant (15-45°)
    - 'y': East direction (90-angle_tol to 90°)
    - '2': SE quadrant (45-90-angle_tol°)

    Vertical channels:
    - 'z': Primary vertical (0-15°)
    - '3': Alternate vertical (15-90°)

    Examples
    --------
    Create north-oriented electric channel::

        >>> code_dict = {
        ...     'component': 'electric',
        ...     'vertical': False,
        ...     'orientation': {'min': 0, 'max': 15}
        ... }
        >>> make_mt_channel(code_dict)
        'ex'

    Create vertical magnetic channel::

        >>> code_dict = {
        ...     'component': 'magnetics',
        ...     'vertical': True,
        ...     'orientation': {'min': 0, 'max': 15}
        ... }
        >>> make_mt_channel(code_dict)
        'hz'
    """

    # Type narrowing: extract and validate component and orientation
    component = code_dict["component"]
    assert isinstance(component, str), "Component must be a string"

    orientation = code_dict["orientation"]
    assert isinstance(orientation, dict), "Orientation must be a dict"

    vertical = code_dict["vertical"]
    assert isinstance(vertical, bool), "Vertical flag must be a bool"

    try:
        mt_comp = mt_code_dict[component]
    except KeyError:
        mt_comp = component

    mt_dir: str = "z"  # Default direction

    if not vertical:
        if orientation["min"] >= 0 and orientation["max"] <= angle_tol:
            mt_dir = "x"
        elif orientation["min"] >= angle_tol and orientation["max"] <= 45:
            mt_dir = "1"
        elif orientation["min"] >= (90 - angle_tol) and orientation["max"] <= 90:
            mt_dir = "y"
        elif orientation["min"] >= 45 and orientation["max"] <= (90 - angle_tol):
            mt_dir = "2"
    else:
        if orientation["min"] >= 0 and orientation["max"] <= angle_tol:
            mt_dir = "z"
        elif orientation["min"] >= angle_tol and orientation["max"] <= 90:
            mt_dir = "3"

    mt_code = f"{mt_comp}{mt_dir}"

    return mt_code
