# -*- coding: utf-8 -*-
"""
Read an amtant.cal file provided by Zonge.  


Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
is a fancy way of saying f x 6 and f x 8. 

 
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import numpy as np

from mt_metadata.timeseries.filters import FrequencyResponseTableFilter
from mt_metadata.utils.mttime import MTime
from mth5.utils.mth5_logger import setup_logger

# =============================================================================
# Variables
# =============================================================================
class CoilResponse:
    def __init__(self, calibration_file=None, angular_frequency=False):

        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
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

    def file_exists(self):
        """
        Check to make sure the file exists

        :return: True if it does, False if it does not
        :rtype: boolean

        """
        if self.calibration_file is None:
            return False

        if self.calibration_file.exists():
            return True
        return False

    def read_antenna_file(self, antenna_calibration_file=None):
        """

        Read in the Antenna file to frequency, amplitude, phase of the proper
        harmonics (6, 8)

        .. note:: Phase is measureed in milli-radians and will be converted
        to radians.

        :param antenna_calibration_file: path to antenna.cal file provided by Zonge
        :type antenna_calibration_file: string or Path

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

    def extrapolate_amplitude(
        self, fap, frequency_limit, order=2, low_cutoff=0.1, high_cutoff=1500
    ):
        """
        Extrapolate amplitude to a frequency limit.

        If the frequency limit determines if extrapolating from the high or low
        side of the response curve.

        Uses an order polynomial in the linear domain to fit the data, 2 works
        well.

        :param fap: DESCRIPTION
        :type fap: TYPE
        :param frequency_limit: DESCRIPTION
        :type frequency_limit: TYPE
        :param order: DESCRIPTION, defaults to 2
        :type order: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if frequency_limit <= low_cutoff:
            f_index = np.where(fap.frequencies <= low_cutoff)
        elif frequency_limit >= high_cutoff:
            f_index = np.where(fap.frequencies >= high_cutoff)
        else:
            raise ValueError(
                "frequency limit is within the pass band, no need to extrapolate"
            )

        x = fap.frequencies[f_index]
        y = fap.amplitudes[f_index]

        return

        # a, b, c = np.polyfit()

    def get_coil_response_fap(self, coil_number):
        """
        Read an amtant.cal file provided by Zonge.


        Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
        is a fancy way of saying f * 6 and f * 8.

        :param coil_number: ANT4 4 digit serial number
        :type coil_number: int or string
        :return: Frequency look up table
        :rtype: :class:`mt_metadata.timeseries.filters.FrequencyResponseTableFilter`

        """

        if self.coil_calibrations is {}:
            self.read_antenna_file(self.calibration_file)

        if self.has_coil_number(coil_number):
            cal = self.coil_calibrations[str(int(coil_number))]
            fap = FrequencyResponseTableFilter()
            fap.frequencies = cal["frequency"]
            fap.amplitudes = cal["amplitude"]
            fap.phases = cal["phase"]
            fap.units_out = "millivolts"
            fap.units_in = "nanotesla"
            fap.name = f"ant4_{coil_number}_response"
            fap.instrument_type = "ANT4 induction coil"
            fap.calibration_date = MTime(
                self.calibration_file.stat().st_mtime
            ).isoformat()

            return fap

        else:
            self.logger.error(
                f"Could not find {coil_number} in {self.calibration_file}"
            )
            raise KeyError(
                f"Could not find {coil_number} in {self.calibration_file}"
            )

    def has_coil_number(self, coil_number):
        """

        Test if coil number is in the antenna file

        :param coil_number: ANT4 serial number
        :type coil_number: int or string
        :return: True if the coil is found, False if it is not
        :rtype: boolean

        """
        if coil_number is None:
            return False

        if self.file_exists():
            coil_number = str(int(float(coil_number)))

            if coil_number in self.coil_calibrations.keys():
                return True
            self.logger.debug(
                f"Could not find {coil_number} in {self.calibration_file}"
            )
            return False
        return False
