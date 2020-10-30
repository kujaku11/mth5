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
    cantainer for a lookup table
    """

    def __init__(self, frequency, filter_values):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.lookup_table = None

        if frequency is not None and filter_values is not None:
            if not isinstance(frequency, np.ndarray):
                frequency = np.array(frequency)
            if not isinstance(filter_values, np.ndarray):
                filter_values = np.array(filter_values)
            if frequency.shape != filter_values.shape:
                msg = (
                    f"filter_values has shape {filter_values.shape} "
                    + f"must have same shape as frequency {frequency.shape}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
            self.lookup_table = np.rec.array(
                [(ff, vv) for ff, vv in zip(frequency, filter_values)],
                dtype=[("frequency", np.float), ("values", np.complex)],
            )

    def __str__(self):
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
                np.arctan2(self.filter_values.real, self.filter_values.imag)
            )
        except AttributeError:
            return None

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
        zeros=None,
        poles=None,
        gain=None,
        frequency=None,
        filter_values=None,
        time_delay=None,
        conversion_factor=None,
    ):

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metadata = metadata.Filter()
        self._zpk = signal.ZerosPolesGain(zeros, poles, gain)

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
