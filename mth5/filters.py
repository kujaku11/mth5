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
        poles=None,
        zeros=None,
        normalization_factor=1.0,
        normalization_frequency=1.0,
        sample_rate=1.0,
    ):

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


class Filter:
    """
    
    Object to hold a filter.
    
    """

    def __init__(self, **kwargs):
        self.metadata = metadata.Filter()
        self.filter = None
        self._poles_zeros = None
        self._lookup_table = None

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
