mth5.utils.fdsn_tools
=====================

.. py:module:: mth5.utils.fdsn_tools

.. autoapi-nested-parse::

   FDSN standards tools.

   Tools for working with FDSN (Incorporated Research Institutions for Seismology)
   standards including SEED channel codes, period/measurement/orientation codes,
   and conversions between SEED and MT (magnetotelluric) channel formats.

   .. rubric:: Notes

   Created on Wed Sep 30 11:47:01 2020

   Author: Jared Peacock

   License: MIT

   .. rubric:: References

   FDSN Channel Codes: https://www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf



Attributes
----------

.. autoapisummary::

   mth5.utils.fdsn_tools.period_code_dict
   mth5.utils.fdsn_tools.measurement_code_dict
   mth5.utils.fdsn_tools.measurement_code_dict_reverse
   mth5.utils.fdsn_tools.orientation_code_dict
   mth5.utils.fdsn_tools.mt_code_dict


Functions
---------

.. autoapisummary::

   mth5.utils.fdsn_tools.get_location_code
   mth5.utils.fdsn_tools.get_period_code
   mth5.utils.fdsn_tools.get_measurement_code
   mth5.utils.fdsn_tools.get_orientation_code
   mth5.utils.fdsn_tools.make_channel_code
   mth5.utils.fdsn_tools.read_channel_code
   mth5.utils.fdsn_tools.make_mt_channel


Module Contents
---------------

.. py:data:: period_code_dict

.. py:data:: measurement_code_dict

.. py:data:: measurement_code_dict_reverse

.. py:data:: orientation_code_dict

.. py:data:: mt_code_dict

.. py:function:: get_location_code(channel_obj: object) -> str

   Generate FDSN location code from channel metadata.

   Creates a 2-character location code from the channel's component
   and channel number.

   :param channel_obj: Channel metadata object with `component` and `channel_number` attributes.
                       Expected to be of type `~mt_metadata.timeseries.Channel`.
   :type channel_obj: object

   :returns: 2-character location code formatted as first letter of component
             and last digit of channel number (e.g., 'E1', 'H0').
   :rtype: str

   .. rubric:: Examples

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


.. py:function:: get_period_code(sample_rate: float) -> str

   Get SEED sampling rate code from sample rate.

   Determines the appropriate FDSN/SEED period code based on the sample
   rate in samples per second. Codes range from 'Q' (highest frequency,
   period < 1 μs) to 'F' (lowest frequency, period 1000-5000 s).

   :param sample_rate: Sample rate in samples per second.
   :type sample_rate: float

   :returns: Single character SEED sampling rate code. Defaults to 'A' if no
             code matches the sample rate.
   :rtype: str

   .. rubric:: Notes

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

   .. rubric:: Examples

   Get code for 100 Hz sample rate::

       >>> get_period_code(100.0)
       'B'

   Get code for 1000 Hz::

       >>> get_period_code(1000.0)
       'D'

   Get code for 10 Hz (default 'A')::

       >>> get_period_code(10.0)
       'M'


.. py:function:: get_measurement_code(measurement: str) -> str

   Get SEED sensor code from measurement type.

   Maps measurement types to single-character SEED sensor codes.
   Performs case-insensitive substring matching.

   :param measurement: Measurement type (e.g., 'electric', 'magnetics', 'temperature',
                       'tilt', 'pressure', 'humidity', 'gravity', 'calibration',
                       'rain_fall', 'water_current', 'wind', 'linear_strain', 'tide',
                       'creep').
   :type measurement: str

   :returns: Single character SEED sensor code. Returns 'Y' if measurement
             type not found in mapping dictionary.
   :rtype: str

   .. rubric:: Notes

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

   .. rubric:: Examples

   Get code for electric measurement::

       >>> get_measurement_code('electric')
       'Q'

   Get code for magnetic field::

       >>> get_measurement_code('magnetics')
       'F'

   Unknown measurement returns 'Y'::

       >>> get_measurement_code('unknown')
       'Y'


.. py:function:: get_orientation_code(azimuth: float, orientation: str = 'horizontal') -> str

   Get SEED orientation code from azimuth and orientation type.

   Maps azimuth angle to SEED orientation code based on whether the
   sensor is oriented horizontally or vertically.

   :param azimuth: Azimuth angle in degrees where 0 is north, 90 is east,
                   180 is south, 270 is west. For vertical orientation,
                   0 = vertical down.
   :type azimuth: float
   :param orientation: Type of sensor orientation.
   :type orientation: {'horizontal', 'vertical'}, default 'horizontal'

   :returns: Single character SEED orientation code.
   :rtype: str

   :raises ValueError: If `orientation` is not 'horizontal' or 'vertical'.

   .. rubric:: Notes

   Horizontal orientation codes (azimuths):
   - 'N': 0-15° (North)
   - 'E': 75-90° (East)
   - '1': 15-45° (NE quadrant)
   - '2': 45-75° (SE quadrant)

   Vertical orientation codes:
   - 'Z': 0-15° (Primary vertical)
   - '3': 15-75° (Alternate vertical)

   .. rubric:: Examples

   Get code for northerly azimuth::

       >>> get_orientation_code(10.0, orientation='horizontal')
       'N'

   Get code for easterly azimuth::

       >>> get_orientation_code(85.0, orientation='horizontal')
       'E'

   Get code for vertical sensor::

       >>> get_orientation_code(0.0, orientation='vertical')
       'Z'


.. py:function:: make_channel_code(channel_obj: object) -> str

   Generate 3-character SEED channel code from channel metadata.

   Combines period code, measurement code, and orientation code into
   a standard 3-character FDSN channel code.

   :param channel_obj: Channel metadata object with attributes: `sample_rate`, `component`,
                       `type`, `measurement_azimuth`, `measurement_tilt`.
                       Expected to be of type `~mt_metadata.timeseries.Channel`.
   :type channel_obj: object

   :returns: 3-character SEED channel code (e.g., 'BHZ', 'HHE').
   :rtype: str

   .. rubric:: Notes

   The channel code format is: [Period Code][Measurement Code][Orientation Code]

   - Period code: based on sample_rate
   - Measurement code: derived from component, with fallback to type
   - Orientation code: depends on whether component is vertical ('z')

   .. rubric:: Examples

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


.. py:function:: read_channel_code(channel_code: str) -> dict[str, dict[str, int] | str | bool]

   Parse FDSN channel code into components.

   Decodes a 3-character SEED channel code into its constituent parts:
   period range, component type, orientation range, and vertical flag.

   :param channel_code: 3-character FDSN channel code (e.g., 'BHZ', 'HHE').
   :type channel_code: str

   :returns: Dictionary with keys:
             - 'period' (dict): Period range with 'min' and 'max' keys (Hz).
             - 'component' (str): Component type (e.g., 'electric', 'magnetics').
             - 'orientation' (dict): Angle range with 'min' and 'max' keys (degrees).
             - 'vertical' (bool): True if component is vertical, False otherwise.
   :rtype: dict

   :raises ValueError: If channel code is not 3 characters, contains invalid period code,
       or contains invalid orientation code.

   .. rubric:: Notes

   Vertical components are identified by orientation codes 'Z' or '3'.

   .. rubric:: Examples

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


.. py:function:: make_mt_channel(code_dict: dict[str, dict[str, int] | str | bool], angle_tol: int = 15) -> str

   Convert FDSN code dictionary to magnetotelluric (MT) channel code.

   Maps FDSN codes to MT channel naming convention (e.g., 'ex', 'hy', 'hz').

   :param code_dict: Dictionary with keys:
                     - 'component' (str): Measurement type (e.g., 'magnetics', 'electric').
                     - 'vertical' (bool): True for vertical, False for horizontal.
                     - 'orientation' (dict): Orientation range with 'min' and 'max'.
   :type code_dict: dict
   :param angle_tol: Angle tolerance in degrees for determining cardinal directions.
   :type angle_tol: int, default 15

   :returns: 2-character MT channel code (e.g., 'ex', 'hy', 'hz').
             Format: [component code][direction code]
             - Component: 'e' (electric) or 'h' (magnetic)
             - Direction: 'x', 'y', 'z' for cardinal or '1', '2', '3' for intermediate
   :rtype: str

   .. rubric:: Notes

   Direction mapping for horizontal channels (0-90°):
   - 'x': North direction (0-15°)
   - '1': NE quadrant (15-45°)
   - 'y': East direction (90-angle_tol to 90°)
   - '2': SE quadrant (45-90-angle_tol°)

   Vertical channels:
   - 'z': Primary vertical (0-15°)
   - '3': Alternate vertical (15-90°)

   .. rubric:: Examples

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


