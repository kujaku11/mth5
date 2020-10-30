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

import pandas as pd
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
                    [(ff, rr + 1j * ii) for ff, rr, ii in zip(frequency, real, imaginary)],
                    dtype=[("frequency", np.float), ("values", np.complex)],
                )

    def __str__(self):
        if self.lookup_table is not None:
            return "\n".join(
                ["frequency   real      imaginary  amplitude   phase"]
                + ["-" * 55]
                + [
                    f"{ff:<12.5g}{vv.real:<10.6g}{vv.imag:<10.6g} {aa:<10.6g}{pp:10.6g}"
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
            msg = (f"Input amplitude and phase must be same shape "
                   + "{amplitude.shape} != {phase.shape}")
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
    
    Object to hold a filter.
    
    """

    def __init__(
        self,
        filter_metadata=None,
        zeros=None,
        poles=None,
        gain=None,
        frequency=None,
        real=None,
        imaginary=None,
        amplitude=None,
        phase=None,
        time_delay=None,
        conversion_factor=None,
    ):

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metadata = metadata.Filter()
        self._zpk = signal.ZerosPolesGain(zeros, poles, gain)
        if amplitude is not None and phase is not None:
            self.lookup_table = LookupTable(frequency, amplitude, phase, amplitude_phase=True)
        else:
            self.lookup_table = LookupTable(frequency, real, imaginary)

        if frequency is not None or filter_values is not None:
            if filter_values is None:
                msg = "Input frequency, must input vvalues as well"
                self.logger.error(msg)
                raise ValueError(msg)
            if frequency is None:
                msg = "Input filter_values, must input vvalues as well"
                self.logger.error(msg)
                raise ValueError(msg)

        self._lookup_table = np.array(np.frequency,)
        self.gain = gain
        self.time_delay = time_delay
        self.conversion_factor = conversion_factor

    @property
    def poles_zeros(self):
        """ Poles and zeros, if None return convesion from lookup table """

        if self._poles_zeros is not None:
            return self._poles_zeros

        if self.lookup_table is not None:
            return self.to_poles_zeros()

        return None

    @property
    def lookup_table(self):
        """ look up table, if None return conversion from poles_zeros """

        if self._lookup_table is not None:
            return self._lookup_table

        if self._poles_zeros is not None:
            return self.to_lookup_table()

        return None

    @poles_zeros.setter
    def poles_zeros(self, pz_array):
        """
        
        set the poles and zeros into a pandas 
        
        :param pz_array: DESCRIPTION
        :type pz_array: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        pass

    @lookup_table.setter
    def lookup_table(self, lookup_array):
        """
        
        :param lookup_array: DESCRIPTION
        :type lookup_array: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
