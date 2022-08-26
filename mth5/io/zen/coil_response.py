# -*- coding: utf-8 -*-
"""
Read an amtant.cal file provided by Zonge.  


Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
is a fancy way of saying f * 6 and f * 8. 

 
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import numpy as np

from mt_metadata.timeseries.filters import FrequencyResponseTableFilter
from mt_metadata.utils.mttime import MTime

# =============================================================================
# Variables
# =============================================================================
class CoilResponse:
    def __init__(self, calibration_file=None, angular_frequency=False):
        self.coil_calibrations = {}
        self._n_frequencies = 48
        self.calibration_file = calibration_file
        self.angular_frequency = angular_frequency
        if calibration_file:
            self.read_antenna_file()

    @property
    def calibration_file(self):
        return self._calibration_fn

    @calibration_file.setter
    def calibration_file(self, fn):
        if fn is not None:
            self._calibration_fn = Path(fn)

        else:
            self._calibration_fn = None

    def read_antenna_file(self, antenna_calibration_file=None):
        """

        Read in the Antenna file to frequency, amplitude, phase of the proper
        harmonics (6, 8)

        .. note:: Phase is measureed in milli-radians and will be converted
        to radians.

        :param antenna_calibration_file: DESCRIPTION
        :type antenna_calibration_file: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if antenna_calibration_file is not None:
            self.calibration_file = antenna_calibration_file

        cal_dtype = [
            ("frequency", float),
            ("amplitude", float),
            ("phase", float),
        ]

        with open(self.calibration_file, "r") as fid:
            lines = fid.readlines()

        self.coil_calibrations = {}
        ff = -2
        for line in lines:
            if "antenna" in line.lower():
                f = float(line.split()[2].strip())
                if self.angular_frequency:
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

    def get_coil_response_fap(self, coil_number):
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

        if self.coil_calibrations is {}:
            self.read_antenna_file(self.calibration_file)
        try:
            cal = self.coil_calibrations[str(int(coil_number))]
            fap = FrequencyResponseTableFilter()
            fap.frequencies = cal["frequency"]
            fap.amplitudes = cal["amplitude"]
            fap.phases = cal["phase"]
            fap.units_in = "millivolts"
            fap.units_out = "nanotesla"
            fap.name = f"coil_{coil_number}"
            fap.instrument_type = "ANT4 induction coil"
            fap.calibration_date = MTime(
                self.calibration_file.stat().st_mtime
            ).isoformat()

            return fap

        except KeyError:
            raise KeyError(f"Could not find {coil_number} in calibration file")
