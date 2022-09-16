# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:54:09 2020

:author: Jared Peacock

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import time
import datetime

import gzip

import numpy as np
import pandas as pd

from mth5 import timeseries
from mth5.io.usgs_ascii import AsciiMetadata

# =============================================================================
#  Metadata for usgs ascii file
# =============================================================================


class USGSasc(AsciiMetadata):
    """
    Read and write USGS ascii formatted time series.

    =================== =======================================================
    Attributes          Description
    =================== =======================================================
    ts                  Pandas dataframe holding the time series data
    fn                  Full path to .asc file
    station_dir         Full path to station directory
    meta_notes          Notes of how the station was collected
    =================== =======================================================

    ============================== ============================================
    Methods                        Description
    ============================== ============================================
    get_z3d_db                     Get Pandas dataframe for schedule block
    locate_mtft24_cfg_fn           Look for a mtft24.cfg file in station_dir
    get_metadata_from_mtft24       Get metadata from mtft24.cfg file
    get_metadata_from_survey_csv   Get metadata from survey csv file
    fill_metadata                  Fill Metadata container from a meta_array
    read_asc_file                  Read in USGS ascii file
    convert_electrics              Convert electric channels to mV/km
    write_asc_file                 Write an USGS ascii file
    write_station_info_metadata    Write metadata to a .cfg file
    ============================== ============================================

    :Example: ::

        >>> zc = Z3DCollection()
        >>> fn_list = zc.get_time_blocks(z3d_path)
        >>> zm = USGSasc()
        >>> zm.SurveyID = 'iMUSH'
        >>> zm.get_z3d_db(fn_list[0])
        >>> zm.read_mtft24_cfg()
        >>> zm.CoordinateSystem = 'Geomagnetic North'
        >>> zm.SurveyID = 'MT'
        >>> zm.write_asc_file(str_fmt='%15.7e')
        >>> zm.write_station_info_metadata()

    """

    def __init__(self, fn=None, **kwargs):
        super().__init__(fn, **kwargs)
        self.ts = None
        self.station_dir = Path().cwd()
        self.meta_notes = None
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    @property
    def hx(self):
        """HX"""
        if self.ts is not None:
            comp_dict = self.get_component_info("hx")
            if comp_dict is None:
                return None
            meta_dict = {
                "channel_number": comp_dict["ChnNum"],
                "component": "hx",
                "measurement_azimuth": comp_dict["Azimuth"],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "magnetic",
                "units": "nanotesla",
                "sensor.id": comp_dict["InstrumentID"],
            }

            return timeseries.MTTS(
                "magnetic",
                data=self.ts.hx.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
            )
        return None

    @property
    def hy(self):
        """hy"""
        if self.ts is not None:
            comp_dict = self.get_component_info("hy")
            if comp_dict is None:
                return None
            meta_dict = {
                "channel_number": comp_dict["ChnNum"],
                "component": "hy",
                "measurement_azimuth": comp_dict["Azimuth"],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "magnetic",
                "units": "nanotesla",
                "sensor.id": comp_dict["InstrumentID"],
            }

            return timeseries.MTTS(
                "magnetic",
                data=self.ts.hy.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
            )
        return None

    @property
    def hz(self):
        """hz"""
        if self.ts is not None:
            comp_dict = self.get_component_info("hz")
            if comp_dict is None:
                return None
            meta_dict = {
                "channel_number": comp_dict["ChnNum"],
                "component": "hz",
                "measurement_azimuth": comp_dict["Azimuth"],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "magnetic",
                "units": "nanotesla",
                "sensor.id": comp_dict["InstrumentID"],
            }

            return timeseries.MTTS(
                "magnetic",
                data=self.ts.hz.to_numpy(),
                channel_metadata={"magnetic": meta_dict},
            )
        return None

    @property
    def ex(self):
        """ex"""
        if self.ts is not None:
            comp_dict = self.get_component_info("ex")
            if comp_dict is None:
                return None
            meta_dict = {
                "channel_number": comp_dict["ChnNum"],
                "component": "ex",
                "measurement_azimuth": comp_dict["Azimuth"],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "electric",
                "units": "millivolts per kilometer",
                "sensor.id": comp_dict["InstrumentID"],
                "dipole_length": comp_dict["Dipole_Length"],
            }

            return timeseries.MTTS(
                "electric",
                data=self.ts.ex.to_numpy(),
                channel_metadata={"electric": meta_dict},
            )
        return None

    @property
    def ey(self):
        """ey"""
        if self.ts is not None:
            comp_dict = self.get_component_info("ey")
            if comp_dict is None:
                return None
            meta_dict = {
                "channel_number": comp_dict["ChnNum"],
                "component": "ey",
                "measurement_azimuth": comp_dict["Azimuth"],
                "measurement_tilt": 0,
                "sample_rate": self.AcqSmpFreq,
                "time_period.start": self.AcqStartTime,
                "time_period.end": self.AcqStopTime,
                "type": "electric",
                "units": "millivolts per kilometer",
                "sensor.id": comp_dict["InstrumentID"],
                "dipole_length": comp_dict["Dipole_Length"],
            }

            return timeseries.MTTS(
                "electric",
                data=self.ts.ey.to_numpy(),
                channel_metadata={"electric": meta_dict},
            )
        return None

    @property
    def electric_channels(self):
        electrics = []
        for key, kdict in self.channel_dict.items():
            if "e" in kdict["ChnID"].lower():
                electrics.append(kdict["ChnID"].lower())

        return ", ".join(electrics)

    @property
    def magnetic_channels(self):
        magnetics = []
        for key, kdict in self.channel_dict.items():
            if "h" in kdict["ChnID"].lower() or "b" in kdict["ChnID"].lower():
                magnetics.append(kdict["ChnID"].lower())

        return ", ".join(magnetics)

    @property
    def run_xarray(self):
        """Get xarray for run"""
        if self.ts is not None:
            meta_dict = {
                "run": {
                    "channels_recorded_electric": self.electric_channels,
                    "channels_recorded_magnetic": self.magnetic_channels,
                    "channels_recorded_auxiliary": None,
                    "comments": self.comments,
                    "id": self.SiteID,
                    "sample_rate": self.sample_rate,
                    "time_period.end": self.AcqStartTime,
                    "time_period.start": self.AcqStopTime,
                }
            }

            return timeseries.RunTS(
                array_list=[self.hx, self.hy, self.hz, self.ex, self.ey],
                run_metadata=meta_dict,
            )

        return None

    def fill_metadata(self, meta_arr):
        """
        Fill in metadata from time array made by
        Z3DCollection.check_time_series.

        :param meta_arr: structured array of metadata for the Z3D files to be
                         combined.
        :type meta_arr: np.ndarray
        """
        try:
            self.AcqNumSmp = self.ts.shape[0]
        except AttributeError:
            pass
        self.AcqSmpFreq = meta_arr["df"].mean()
        self.AcqStartTime = meta_arr["start"].max()
        self.AcqStopTime = meta_arr["stop"].min()
        try:
            self.Nchan = self.ts.shape[1]
        except AttributeError:
            self.Nchan = meta_arr.shape[0]
        self.RunID = 1
        self.SiteLatitude = np.median(meta_arr["lat"])
        self.SiteLongitude = np.median(meta_arr["lon"])
        fn = Path((meta_arr["fn"][0]))
        self.SiteID = fn.stem
        self.station_dir = fn.parent

        # if geographic coordinates add in declination
        if "geographic" in self.CoordinateSystem.lower():
            meta_arr["ch_azimuth"][
                np.where(meta_arr["comp"] != "hz")
            ] += self.declination

        # fill channel dictionary with appropriate values
        self.channel_dict = dict(
            [
                (
                    comp.capitalize(),
                    {
                        "ChnNum": "{0}{1}".format(self.SiteID, ii + 1),
                        "ChnID": meta_arr["comp"][ii].capitalize(),
                        "InstrumentID": meta_arr["ch_box"][ii],
                        "Azimuth": meta_arr["ch_azimuth"][ii],
                        "Dipole_Length": meta_arr["ch_length"][ii],
                        "n_samples": meta_arr["n_samples"][ii],
                        "n_diff": meta_arr["t_diff"][ii],
                        "std": meta_arr["std"][ii],
                        "start": meta_arr["start"][ii],
                    },
                )
                for ii, comp in enumerate(meta_arr["comp"])
            ]
        )
        for ii, comp in enumerate(meta_arr["comp"]):
            if "h" in comp.lower():
                self.channel_dict[comp.capitalize()][
                    "InstrumentID"
                ] += "-{0}".format(meta_arr["ch_num"])

    def read_asc_file(self, fn=None):
        """
        Read in a USGS ascii file and fill attributes accordingly.

        :param fn: full path to .asc file to be read in
        :type fn: string
        """
        if fn is not None:
            self.fn = fn
        st = datetime.datetime.now()
        data_line = self.read_metadata()
        self.ts = pd.read_csv(
            self.fn,
            delim_whitespace=True,
            skiprows=data_line,
            dtype=np.float32,
        )
        dt_freq = "{0:.0f}N".format(1.0 / (self.AcqSmpFreq) * 1e9)
        dt_index = pd.date_range(
            start=self.AcqStartTime, periods=self.AcqNumSmp, freq=dt_freq
        )
        self.ts.index = dt_index
        self.ts.columns = self.ts.columns.str.lower()

        et = datetime.datetime.now()
        read_time = et - st
        self.logger.info("Reading took {0}".format(read_time.total_seconds()))

    def _make_file_name(
        self, save_path=None, compression=True, compress_type="zip"
    ):
        """
        get the file name to save to

        :param save_path: full path to directory to save file to
        :type save_path: string

        :param compression: compress file
        :type compression: [ True | False ]

        :return: save_fn
        :rtype: string

        """
        # make the file name to save to
        if save_path is not None:
            save_path = Path(save_path)
            save_fn = save_path.joinpath(
                "{0}_{1}T{2}_{3:.0f}.asc".format(
                    self.SiteID,
                    self._start_time.strftime("%Y-%m-%d"),
                    self._start_time.strftime("%H%M%S"),
                    self.AcqSmpFreq,
                ),
            )
        else:
            save_fn = self.station_dir.joinpath(
                "{0}_{1}T{2}_{3:.0f}.asc".format(
                    self.SiteID,
                    self._start_time.strftime("%Y-%m-%d"),
                    self._start_time.strftime("%H%M%S"),
                    self.AcqSmpFreq,
                ),
            )

        if compression:
            if compress_type == "zip":
                save_fn = save_fn + ".zip"
            elif compress_type == "gzip":
                save_fn = save_fn + ".gz"

        return save_fn

    def write_asc_file(
        self,
        save_fn=None,
        chunk_size=1024,
        str_fmt="%15.7e",
        full=True,
        compress=False,
        save_dir=None,
        compress_type="zip",
        convert_electrics=True,
    ):
        """
        Write an ascii file in the USGS ascii format.

        :param save_fn: full path to file name to save the merged ascii to
        :type save_fn: string

        :param chunck_size: chunck size to write file in blocks, larger numbers
                            are typically slower.
        :type chunck_size: int

        :param str_fmt: format of the data as written
        :type str_fmt: string

        :param full: write out the complete file, mostly for testing.
        :type full: boolean [ True | False ]

        :param compress: compress file
        :type compress: boolean [ True | False ]

        :param compress_type: compress file using zip or gzip
        :type compress_type: boolean [ zip | gzip ]
        """
        # get the filename to save to
        save_fn = self._make_file_name(
            save_path=save_dir,
            compression=compress,
            compress_type=compress_type,
        )
        # get the number of characters in the desired string
        s_num = int(str_fmt[1 : str_fmt.find(".")])

        # convert electric fields into mV/km
        if convert_electrics:
            self.convert_electrics()

        self.logger.debug("==> {0}".format(save_fn))
        self.logger.debug("START --> {0}".format(time.ctime()))
        st = datetime.datetime.now()

        # write meta data first
        # sort channel information same as columns
        meta_lines = self.write_metadata(
            chn_list=[c.capitalize() for c in self.ts.columns]
        )
        if compress == True and compress_type == "gzip":
            with gzip.open(save_fn, "wb") as fid:
                h_line = [
                    "".join(
                        [
                            "{0:>{1}}".format(c.capitalize(), s_num)
                            for c in self.ts.columns
                        ]
                    )
                ]
                fid.write("\n".join(meta_lines + h_line) + "\n")

                # write out data
                if full is False:
                    out = np.array(self.ts[0:chunk_size])
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = "\n".join(
                        ["".join(out[ii, :]) for ii in range(out.shape[0])]
                    )
                    fid.write(lines + "\n")
                    self.logger.debug("END --> {0}".format(time.ctime()))
                    et = datetime.datetime.now()
                    write_time = et - st
                    self.logger.debug(
                        "Writing took: {0} seconds".format(
                            write_time.total_seconds()
                        )
                    )
                    return

                for chunk in range(int(self.ts.shape[0] / chunk_size)):
                    out = np.array(
                        self.ts[chunk * chunk_size : (chunk + 1) * chunk_size]
                    )
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = "\n".join(
                        ["".join(out[ii, :]) for ii in range(out.shape[0])]
                    )
                    fid.write(lines + "\n")

        else:
            if compress == True and compress_type == "zip":
                self.logger.debug("ZIPPING")
                save_fn = save_fn[0:-4]
                zip_file = True
                self.logger.debug(zip_file)
            with open(save_fn, "w") as fid:
                h_line = [
                    "".join(
                        [
                            "{0:>{1}}".format(c.capitalize(), s_num)
                            for c in self.ts.columns
                        ]
                    )
                ]
                fid.write("\n".join(meta_lines + h_line) + "\n")

                # write out data
                if full is False:
                    out = np.array(self.ts[0:chunk_size])
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = "\n".join(
                        ["".join(out[ii, :]) for ii in range(out.shape[0])]
                    )
                    fid.write(lines + "\n")
                    self.logger.debug("END --> {0}".format(time.ctime()))
                    et = datetime.datetime.now()
                    write_time = et - st
                    self.logger.debug(
                        "Writing took: {0} seconds".format(
                            write_time.total_seconds()
                        )
                    )
                    return

                for chunk in range(int(self.ts.shape[0] / chunk_size)):
                    out = np.array(
                        self.ts[chunk * chunk_size : (chunk + 1) * chunk_size]
                    )
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = "\n".join(
                        ["".join(out[ii, :]) for ii in range(out.shape[0])]
                    )
                    fid.write(lines + "\n")

        # for some fucking reason, all interal variables don't exist anymore
        # and if you try to do the zipping nothing happens, so have to do
        # it externally.  WTF
        self.logger.debug("END -->   {0}".format(time.ctime()))
        et = datetime.datetime.now()
        write_time = et - st
        self.logger.debug(
            "Writing took: {0} seconds".format(write_time.total_seconds())
        )


def read_ascii(fn):
    """
    read USGS ASCII formatted file

    :param fn: DESCRIPTION
    :type fn: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    asc_obj = USGSasc(fn)
    asc_obj.read_asc_file()

    return asc_obj.run_xarray
