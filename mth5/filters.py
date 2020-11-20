# -*- coding: utf-8 -*-
"""

Filter object

Created on Wed Sep 30 12:55:58 2020

:author: Jared Peacock

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy import signal
import logging

from mth5 import metadata

# =============================================================================
#
# =============================================================================
class PolesZeros:
    """ 
    
    Container to hold poles and zeros
    
    """

    def __init__(
        self,
        poles=[],
        zeros=[],
        normalization_factor=1.0,
        normalization_frequency=1.0,
        sample_rate=1.0,
    ):

        self._zpk = signal.ZerosPolesGain(zeros, poles, normalization_factor)
        self._poles = None
        self._zeros = None
        self._normalization_factor = None
        self._normalization_frequency = None
        self._sample_rate = None

        self.poles = poles
        self.zeros = zeros
        self.normalization_factor = normalization_factor
        self.normalization_frequency = normalization_frequency
        self.sample_rate = sample_rate

    def __str__(self):
        return "\n".join(
            [
                "Poles and Zeros Filter:",
                f"\tNumber of Poles: {self.n_poles}",
                f"\tNumber of Zeros: {self.n_zeros}",
                f"\tPoles = {self.poles}",
                f"\tZeros = {self.zeros}",
                f"\tNormalization Factor = {self.normalization_factor}",
                f"\tNormalization Frequency = {self.normalization_frequency}",
            ]
        )

    def __repr__(self):
        return self.__str__()

    @property
    def poles(self):
        return self._poles

    @poles.setter
    def poles(self, value):
        """
        Make sure that any input is set to an np.ndarray with dtype complex
        
        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(value, (int, float, complex)):
            self._poles = np.array([value], dtype=np.complex)

        if isinstance(value, (tuple, list, np.ndarray)):
            self._poles = np.array(value, dtype=np.complex)

    @property
    def zeros(self):
        return self._zeros

    @zeros.setter
    def zeros(self, value):
        """
        Make sure that any input is set to an np.ndarray with dtype complex
        
        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(value, (int, float, complex)):
            self._zeros = np.array([value], dtype=np.complex)

        if isinstance(value, (tuple, list, np.ndarray)):
            self._zeros = np.array(value, dtype=np.complex)

    @property
    def normalization_factor(self):
        return self._normalization_factor

    @normalization_factor.setter
    def normalization_factor(self, value):
        self._normalization_factor = float(value)

    @property
    def normalization_frequency(self):
        return self._normalization_frequency

    @normalization_frequency.setter
    def normalization_frequency(self, value):
        self._normalization_frequency = float(value)

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    @property
    def n_poles(self):
        if self.poles is not None:
            return int(self.poles.size)
        return 0

    @property
    def n_zeros(self):
        if self.zeros is not None:
            return int(self.zeros.size)
        return 0

    def to_lookup_table(self, frequencies=None):
        """
        compute the look up table from the poles and zeros
        
        :param frequencies: DESCRIPTION, defaults to None
        :type frequencies: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        pz = signal.ZerosPolesGain(
            self.zeros, self.poles, self.normalization_factor, dt=self.sample_rate
        )

        f, amp = signal.dfreqresp(pz, frequencies)

        return f, amp


class LookupTable:
    """
    cantainer for a lookup table of frequency response filter.
    
    Commonly measured as frequency, real, imaginary but can be amplitude and phase
    
        
    :param frequency: frequencies at which the filter response is measured at
    :type frequency: list, tuple, np.ndarray
    
    :param real: real parts of the filter response, needs to be same size as 
    frequency.  If amplitude_phase is True real is the amplitude
    :type real: list, tuple, np.ndarray
    
    :param imaginary: imaginary parts of the filter response, needs to be same size as 
    frequency.  If amplitude_phase is True imaginary is the phase
    :type imaginary: list, tuple, np.ndarray
    
    :param amplitude_phase: True if the inputs are amplitude and phase, defaults to
    False
    :type amplitude_phase: Boolean, optional
    
    :raises ValueError:  

    >>> from mth5.filters import LookupTable
    
    Initialize an empty filter
    
    >>> response = LookupTable(None, None, None)
    
    Fill it with values
    
    >>> response.frequency = np.logspace(-3, 3, 50)
    >>> response.filter_values = np.random.rand(50) + 1j * np.random.rand(50)
    
    Input amplitude and phase
    
    >>> freq = np.logspace(-3, 3, 50)
    >>> amp = np.random.rand(50)
    >>> phase = 180 * np.random.rand(50)
    >>> response = LookupTable(freq, amp, phase, amplitude_phase=True)
    
    
    """

    def __init__(self, frequency, real, imaginary, amplitude_phase=False):

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.lookup_table = None

        if frequency is not None and real is not None and imaginary is not None:
            if not isinstance(frequency, np.ndarray):
                frequency = np.array(frequency)
            if not isinstance(real, np.ndarray):
                real = np.array(real)
            if not isinstance(imaginary, np.ndarray):
                imaginary = np.array(imaginary)
            if frequency.shape != real.shape or real.shape != imaginary.shape:
                msg = (
                    f"filter_values has shape {real.shape} "
                    + f"must have same shape as frequency {frequency.shape}"
                )
                self.logger.error(msg)
                raise ValueError(msg)

            if amplitude_phase:
                self.frequency = frequency
                self.from_amplitude_phase(real, imaginary)
            else:
                self.lookup_table = np.rec.array(
                    [
                        (ff, rr + 1j * ii)
                        for ff, rr, ii in zip(frequency, real, imaginary)
                    ],
                    dtype=[("frequency", np.float), ("values", np.complex)],
                )

    def __str__(self):
        if self.lookup_table is not None:
            return "\n".join(
                [
                    "".join(
                        [
                            f"{vv:<14}"
                            for vv in [
                                "frequency",
                                "real",
                                "imaginary",
                                "amplitude",
                                "phase",
                            ]
                        ]
                    )
                ]
                + ["-" * 70]
                + [
                    f"{ff:<14.5e}{vv.real:<14.5e}{vv.imag:<14.5e}{aa:<14.5e}{pp:<14.5e}"
                    for ff, vv, aa, pp in zip(
                        self.frequency, self.filter_values, self.amplitude, self.phase
                    )
                ]
            )
        return "No filter data"

    def __repr__(self):
        return self.__str__()

    @property
    def frequency(self):
        try:
            return self.lookup_table.frequency
        except AttributeError:
            return None

    @frequency.setter
    def frequency(self, values):
        if self.lookup_table is None:
            self.lookup_table = np.rec.array(
                [(ff, None) for ff in values],
                dtype=[("frequency", np.float), ("values", np.complex)],
            )
        if len(values) != len(self.frequency):
            msg = f"new values have length {len(values)}, must be {len(self.frequency)}"
            self.logger.error(msg)
            raise ValueError(msg)

        self.lookup_table.frequency = values

    @property
    def filter_values(self):
        try:
            return self.lookup_table.values
        except AttributeError:
            return None

    @filter_values.setter
    def filter_values(self, values):
        if self.lookup_table is None:
            self.lookup_table = np.rec.array(
                [(None, ff) for ff in values],
                dtype=[("frequency", np.float), ("values", np.complex)],
            )
        if len(values) != len(self.filter_values):
            msg = f"new values have length {len(values)}, must be {len(self.frequency)}"
            self.logger.error(msg)
            raise ValueError(msg)

        self.lookup_table.values = values

    @property
    def amplitude(self):
        try:
            return np.sqrt(self.filter_values.real ** 2 + self.filter_values.imag ** 2)
        except AttributeError:
            return None

    @property
    def phase(self):
        try:
            return np.rad2deg(
                np.arctan2(self.filter_values.imag, self.filter_values.real)
            )
        except AttributeError:
            return None

    def from_amplitude_phase(self, amplitude, phase):
        """ 
        compute real and imaginary from amplitude and phase for an existing
        or new frequency range.
        
        :param amplitude: amplitude values
        :type amplitude: list, tuple, np.ndarray
        
        :param phase: phase angle in degrees
        :type phase: list, tuple, np.ndarray
        
        converts to real and imaginary
        
        """

        if not isinstance(amplitude, np.ndarray):
            amplitude = np.array(amplitude)

        if not isinstance(phase, np.ndarray):
            phase = np.array(phase)

        if not amplitude.shape == phase.shape:
            msg = (
                "Input amplitude and phase must be same shape "
                + f"{amplitude.shape} != {phase.shape}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        real = amplitude * np.cos(np.deg2rad(phase))
        imag = amplitude * np.sin(np.deg2rad(phase))

        self.filter_values = real + 1j * imag

    def to_poles_zeros(self):
        """
        Convert lookup table to poles and zeros
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        pass


class Filter:
    """
    
    All encompassing filter container.  Will hold filters:
        * Poles and Zeros (zpk)
        * Look up table (lookup_table)
        * Conversion (conversion_factor)
        * Time delay (time_delay)
        * Gain (gain) 
        
    .. note:: Gain is for an instrument gain only, the poles and zeros gain is 
    containted within the zpk attribute.
        
    This is just a container for easier access to the filter properties, the actual
    application of the filter should be done by the timeseries object.
    
    There will be an estimation of poles and zeros to a lookup table and back.
    
    The conversion from zeros and poles to a look up table is trivial
    
    >>> from mth5 import filters
    >>> f = filters.Filter(zeros=[1, 5], poles=[2, 3], gain=5)
    >>> f.zpk.freqresp(np.logspace(-3, 3, 50))
    
    However, the conversion from a look up table to poles and zeros is not trivial
    and requires fitting the frequency response to a transfer function.  And this
    still needs to be implemented.
    
    """

    def __init__(
        self,
        filter_metadata=None,
        zeros=None,
        poles=None,
        zpk_gain=None,
        frequency=None,
        real=None,
        imaginary=None,
        amplitude=None,
        phase=None,
        time_delay=None,
        conversion_factor=None,
        instrument_gain=None,
    ):

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metadata = metadata.Filter()
        if zeros is None:
            zeros = []
        if poles is None:
            poles = []
        if zpk_gain is None:
            zpk_gain = []
        self.zpk = signal.ZerosPolesGain(zeros, poles, zpk_gain)

        if filter_metadata is not None:
            if not isinstance(filter_metadata, metadata.Filter):
                msg = (
                    "Input metadata must be type metadata.Filter not "
                    + f"{type(filter_metadata)}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
            self.metadata.from_dict(filter_metadata.to_dict(required=False))

        # initiate lookup table
        if amplitude is not None and phase is not None:
            self.lookup_table = LookupTable(
                frequency, amplitude, phase, amplitude_phase=True
            )
        else:
            self.lookup_table = LookupTable(frequency, real, imaginary)

        self.instrument_gain = instrument_gain
        self.time_delay = time_delay
        self.conversion_factor = conversion_factor

    def __str__(self):
        info = self.metadata.__str__()
        if self.conversion_factor is not None:
            info += "\n".join(
                ["", "Conversion Factor:", f"\t{self.conversion_factor:<12.5e}"]
            )
        if self.instrument_gain is not None:
            info += "\n".join(
                ["", "Instrument Gain:", f"\t{self.instrument_gain:<12.5e}"]
            )
        if self.time_delay is not None:
            info += "\n".join(["", "Time Delay:", f"\t{self.time_delay:<12.5e}"])

        if len(self.zpk.poles) > 0 or len(self.zpk.zeros) > 0:
            pz = [""]
            pz += ["Poles:"] + [f"\t{vv:<12.5e}" for vv in self.zpk.poles]
            pz += ["Zeros:"] + [f"\t{vv:<12.5e}" for vv in self.zpk.zeros]
            pz += ["Gain:"] + [f"\t{self.zpk.gain:<12.5e}"]
            info += "\n".join(pz)

        if self.lookup_table.frequency is not None:
            info += self.lookup_table.__str__()

        return info

    def __repr__(self):
        return self.__str__()

    def zpk_to_lookup_table(self, frequency):
        """
        convert Poles and Zeros to a look up table of the given frequencies
        
        :param frequency: frequencies to estimate the response
        :type frequency: list, tuple, np.ndarray
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(frequency, np.ndarray):
            frequency = np.array(frequency)

        try:
            f, values = self.zpk.freqresp(w=frequency)
            self.lookup_table = LookupTable(f, values.real, values.imag)
        except ValueError:
            msg = "No zero, poles, gain information to estimate frequency response"
            self.logger.warning(msg)

    def lookup_table_to_zpk(self, n_poles=None, n_zeros=None):
        """
        Estimate poles, zeros, and gain from a lookup table
        
        If an estimate of the number of poles or zeros can help constrain the 
        inversion that is a good thing.
        
        :param n_poles: DESCRIPTION, defaults to None
        :type n_poles: TYPE, optional
        :param n_zeros: DESCRIPTION, defaults to None
        :type n_zeros: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

    def from_xml(self, xml_string):
        """
        Read Anna's XML filter files, it needs to be split into strings first.
        
        :param xml_fn: DESCRIPTION
        :type xml_fn: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
