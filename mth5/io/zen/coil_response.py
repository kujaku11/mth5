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
from typing import Any

import numpy as np
from loguru import logger
from mt_metadata.common.mttime import MTime
from mt_metadata.timeseries.filters import FrequencyResponseTableFilter


# =============================================================================
# Variables
# =============================================================================
class CoilResponse:
    """Read ANT4 coil calibration files from Zonge (``amtant.cal``).

    This class parses a Zonge antenna calibration file and exposes a
    :class:`mt_metadata.timeseries.filters.FrequencyResponseTableFilter` for a
    specified coil number.

    Parameters
    ----------
    calibration_file : str | Path | None, optional
        Path to the antenna calibration file. If provided the file will be
        read during initialization, by default None.
    angular_frequency : bool, optional
        If True, reported frequencies will be converted to angular frequency
        (rad/s), by default False.

    Attributes
    ----------
    coil_calibrations : dict[str, numpy.ndarray]
        Mapping of coil serial numbers to a structured numpy array containing
        frequency, amplitude, and phase columns.

    Examples
    --------
    >>> from mth5.mth5.io.zen.coil_response import CoilResponse
    >>> cr = CoilResponse('amtant.cal')
    >>> fap = cr.get_coil_response_fap(1234)
    >>> print(fap.name)

    """

    def __init__(
        self,
        calibration_file: str | Path | None = None,
        angular_frequency: bool = False,
    ) -> None:
        self.logger = logger
        self.coil_calibrations: dict[str, np.ndarray] = {}
        self._n_frequencies: int = 48
        self.calibration_file = calibration_file
        self.angular_frequency: bool = angular_frequency
        if calibration_file:
            # defer to read_antenna_file which handles path coercion
            self.read_antenna_file()
        self._extrapolate_values: dict[str, dict[str, Any]] = {
            "low": {"frequency": 1e-10, "amplitude": 1e-8, "phase": np.pi / 2},
            "high": {"frequency": 1e5, "amplitude": 1e-4, "phase": np.pi / 6},
        }
        self._low_frequency_cutoff: int = 250

    @property
    def calibration_file(self):
        return self._calibration_fn

    @calibration_file.setter
    def calibration_file(self, fn: str | Path | None):
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
        return self.calibration_file.exists()

    def read_antenna_file(
        self, antenna_calibration_file: str | Path | None = None
    ) -> None:
        """Read a Zonge antenna calibration file and parse coil responses.

        The expected file format contains blocks starting with an "antenna"
        header line that provides the base frequency followed by lines with
        coil serial number and amplitude/phase values for the 6th and 8th
        harmonics.

        Parameters
        ----------
        antenna_calibration_file : str | Path | None, optional
            Optional path to the antenna calibration file. If provided, it
            overrides the instance ``calibration_file``.

        Notes
        -----
        Phase values in the file are expected in milliradians and are
        converted to radians.
        """

        self.coil_calibrations = {}
        if antenna_calibration_file is not None:
            self.calibration_file = antenna_calibration_file
        if self.calibration_file is None:
            self.logger.error("No calibration file provided")
            return
        cal_dtype = [
            ("frequency", float),
            ("amplitude", float),
            ("phase", float),
        ]

        with open(self.calibration_file, "r") as fid:
            lines = fid.readlines()

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
                phase6 = float(line_list[2]) / 1000.0
                amp8 = float(line_list[3])
                phase8 = float(line_list[4]) / 1000.0

                if ant not in self.coil_calibrations:
                    self.coil_calibrations[ant] = np.zeros(
                        self._n_frequencies, dtype=cal_dtype
                    )

                self.coil_calibrations[ant][ff] = (f * 6, amp6, phase6)
                self.coil_calibrations[ant][ff + 1] = (f * 8, amp8, phase8)

    def get_coil_response_fap(
        self, coil_number: int | str, extrapolate: bool = True
    ) -> FrequencyResponseTableFilter:
        """
        Read an amtant.cal file provided by Zonge.


        Apparently, the file includes the 6th and 8th harmonic of the given frequency, which
        is a fancy way of saying f * 6 and f * 8.

        :param coil_number: ANT4 4 digit serial number
        :type coil_number: int or string
        :return: Frequency look up table
        :rtype: :class:`mt_metadata.timeseries.filters.FrequencyResponseTableFilter`

        """

        # ensure calibrations are loaded
        if not self.coil_calibrations:
            self.read_antenna_file(self.calibration_file)

        if self.has_coil_number(coil_number):
            cal = self.coil_calibrations[str(int(coil_number))]
            fap = FrequencyResponseTableFilter()
            fap.frequencies = cal["frequency"]
            fap.amplitudes = cal["amplitude"]
            fap.phases = cal["phase"]
            fap.units_out = "milliVolt"
            fap.units_in = "nanoTesla"
            fap.name = f"ant4_{coil_number}_response"
            fap.instrument_type = "ANT4 induction coil"
            fap.calibration_date = MTime(
                time_stamp=self.calibration_file.stat().st_mtime
            ).isoformat()

            if extrapolate:
                return self.extrapolate(fap)
            return fap

        self.logger.error(f"Could not find {coil_number} in {self.calibration_file}")
        raise KeyError(f"Could not find {coil_number} in {self.calibration_file}")

    def extrapolate(
        self, fap: FrequencyResponseTableFilter
    ) -> FrequencyResponseTableFilter:
        """Extrapolate a frequency/amplitude/phase table using log-linear pads.

        Parameters
        ----------
        fap : FrequencyResponseTableFilter
            Frequency response object to extrapolate.

        Returns
        -------
        FrequencyResponseTableFilter
            A copy of ``fap`` with low- and high-frequency extrapolated
            values appended.
        """

        if self._low_frequency_cutoff is not None:
            index = np.where(fap.frequencies < 1.0 / self._low_frequency_cutoff)[0][-1]
        else:
            index = 0

        new_fap = fap.copy()
        new_fap.frequencies = np.append(
            np.append(
                [self._extrapolate_values["low"]["frequency"]], fap.frequencies[index:]
            ),
            self._extrapolate_values["high"]["frequency"],
        )
        new_fap.amplitudes = np.append(
            np.append(
                [self._extrapolate_values["low"]["amplitude"]], fap.amplitudes[index:]
            ),
            self._extrapolate_values["high"]["amplitude"],
        )
        new_fap.phases = np.append(
            np.append([self._extrapolate_values["low"]["phase"]], fap.phases[index:]),
            self._extrapolate_values["high"]["phase"],
        )

        return new_fap

    def has_coil_number(self, coil_number: int | str | None) -> bool:
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
