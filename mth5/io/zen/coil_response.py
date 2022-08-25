# -*- coding: utf-8 -*-
"""
Read an amtant.cal file provided by Zonge.  


Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
is a fancy way of saying f * 6 and f * 8. 

variables
-----------

    **ant_fn**: full path to the calibration file
    
    **birrp**: If the calibration files are written for BIRRP then need to add
    a line at the beginning of the file that describes any scaling factors for 
    the calibrations, should be 1, 1, 1
    
    **angular_frequency**: Puts the frequency in angular frequency (2 * pi * f)
     
    **quadrature**: puts the response in amplitude and phase (True) or 
    real and imaginary (False)
    
    **nf**: number of expected frequencies
 
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import numpy as np
import pandas as pd

from mt_metadata.timeseries.filters import FrequencyResponseTableFilter

# =============================================================================
# Variables
# =============================================================================
class CoilResponse:
    def __init__(self):
        self.coil_calibrations = {}
        self._n_frequencies = 48

    def read_antenna_file(
        self, antenna_calibration_file, angular_frequency=False
    ):
        """

        :param antenna_calibration_file: DESCRIPTION
        :type antenna_calibration_file: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        ant_fn = Path(antenna_calibration_file)

        cal_dtype = [
            ("frequency", np.float),
            ("amplitude", np.float),
            ("phase", np.float),
        ]

        with open(ant_fn, "r") as fid:
            lines = fid.readlines()

        self.coil_calibrations = {}
        ff = -2
        for line in lines:
            if "antenna" in line.lower():
                f = float(line.split()[2].strip())
                if angular_frequency:
                    f = 2 * np.pi * f

                ff += 2
            elif len(line.strip().split()) == 0:
                continue
            else:
                line_list = line.strip().split()
                ant = line_list[0]
                amp6 = float(line_list[1])
                phase6 = float(line_list[2]) / 1000
                amp8 = float(line_list[3])
                phase8 = float(line_list[4]) / 1000

                try:
                    self.coil_calibrations[ant]
                except KeyError:
                    self.coil_calibrations[ant] = np.zeros(
                        self._n_frequencies, dtype=cal_dtype
                    )

                self.coil_calibrations[ant][ff] = (f * 6, amp6, phase6)
                self.coil_calibrations[ant][ff + 1] = (f * 8, amp8, phase8)

    def get_coil_response_fap(self, antenna_calibration_file, coil_number):
        """
        Read an amtant.cal file provided by Zonge.


        Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
        is a fancy way of saying f * 6 and f * 8.

        :param antenna_calibration_file: DESCRIPTION
        :type antenna_calibration_file: TYPE
        :param coil_num: DESCRIPTION
        :type coil_num: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        self.read_antenna_file(antenna_calibration_file)
        try:
            cal = self.coil_calibrations[str(int(coil_number))]
            fap = FrequencyResponseTableFilter()
            fap.frequencies = cal["frequency"]
            fap.amplitudes = cal["amplitude"]
            fap.phases = cal["phase"]
            fap.units_in = "millivolts"
            fap.units_out = "nanotesla"
            fap.name = f"coil_{coil_number}"

            return fap

        except KeyError:
            raise KeyError(f"Could not find {coil_number} in calibration file")
