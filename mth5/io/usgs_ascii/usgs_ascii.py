# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:54:09 2020

:author: Jared Peacock

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================

import datetime
import gzip
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mth5.io.usgs_ascii import AsciiMetadata
from mth5.timeseries import ChannelTS, RunTS


# =============================================================================
#  Metadata for usgs ascii file
# =============================================================================


class USGSascii(AsciiMetadata):
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

    ### need copy in the metadata otherwise the order in the channels
    ### gets messed up

    @property
    def hx(self):
        """HX"""
        if self.ts is not None:
            return ChannelTS(
                "magnetic",
                data=self.ts.hx.to_numpy(),
                channel_metadata=self.hx_metadata.copy(),
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                survey_metadata=self.survey_metadata.copy(),
            )
        return None

    @property
    def hy(self):
        """hy"""
        if self.ts is not None:
            return ChannelTS(
                "magnetic",
                data=self.ts.hy.to_numpy(),
                channel_metadata=self.hy_metadata.copy(),
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                survey_metadata=self.survey_metadata.copy(),
            )
        return None

    @property
    def hz(self):
        """hz"""
        if self.ts is not None:
            return ChannelTS(
                "magnetic",
                data=self.ts.hz.to_numpy(),
                channel_metadata=self.hz_metadata.copy(),
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                survey_metadata=self.survey_metadata.copy(),
            )
        return None

    @property
    def ex(self):
        """ex"""
        if self.ts is not None:
            return ChannelTS(
                "electric",
                data=self.ts.ex.to_numpy(),
                channel_metadata=self.ex_metadata.copy(),
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                survey_metadata=self.survey_metadata.copy(),
            )
        return None

    @property
    def ey(self):
        """ey"""
        if self.ts is not None:
            return ChannelTS(
                "electric",
                data=self.ts.ey.to_numpy(),
                channel_metadata=self.ey_metadata.copy(),
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                survey_metadata=self.survey_metadata.copy(),
            )
        return None

    def to_run_ts(self):
        """Get xarray for run"""
        if self.ts is not None:
            return RunTS(
                array_list=[self.hx, self.hy, self.hz, self.ex, self.ey],
                run_metadata=self.run_metadata.copy(),
                station_metadata=self.station_metadata.copy(),
                survey_metadata=self.survey_metadata.copy(),
            )
        return None

    def read(self, fn=None):
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
            sep="\s+",
            skiprows=data_line,
            dtype=np.float32,
        )
        dt_index = pd.date_range(
            start=self.start.time_stamp, periods=self.n_samples, end=self.end.time_stamp
        )
        self.ts.index = dt_index
        self.ts.columns = self.ts.columns.str.lower()

        et = datetime.datetime.now()
        read_time = et - st
        self.logger.info(f"Reading took {read_time.total_seconds()}")

    def _make_file_name(self, save_path=None, compression=True, compress_type="zip"):
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
            # Use conventional suffixes: zip -> .zip, gzip -> .gz
            if compress_type == "zip":
                save_fn = save_fn.with_suffix(save_fn.suffix + ".zip")
            elif compress_type == "gzip":
                save_fn = save_fn.with_suffix(save_fn.suffix + ".gz")
        return save_fn

    def write(
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
        self.logger.debug(f"==> {save_fn}")
        self.logger.debug("START --> {time.ctime()}")
        st = datetime.datetime.now()

        # write meta data first
        # sort channel information same as columns
        meta_lines = self.write_metadata(
            chn_list=[c.capitalize() for c in self.ts.columns]
        )
        if compress == True and compress_type == "gzip":
            # gzip expects bytes; encode strings before writing
            with gzip.open(save_fn, "wb") as fid:
                h_line = [
                    "".join(
                        [
                            "{0:>{1}}".format(c.capitalize(), s_num)
                            for c in self.ts.columns
                        ]
                    )
                ]
                header_bytes = ("\n".join(meta_lines + h_line) + "\n").encode("utf-8")
                fid.write(header_bytes)

                # write out data
                if full is False:
                    out = np.array(self.ts[0:chunk_size])
                    out[np.where(out == 0)] = float(self.MissingDataFlag)
                    out = np.char.mod(str_fmt, out)
                    lines = "\n".join(
                        ["".join(out[ii, :]) for ii in range(out.shape[0])]
                    )
                    fid.write((lines + "\n").encode("utf-8"))
                    self.logger.debug(f"END --> {time.ctime()}")
                    et = datetime.datetime.now()
                    write_time = et - st
                    self.logger.debug(
                        f"Writing took: {write_time.total_seconds()} seconds"
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
                    fid.write((lines + "\n").encode("utf-8"))
        else:
            if compress == True and compress_type == "zip":
                # Create a temporary .asc file, then zip it into the final
                # .zip archive. `save_fn` is expected to be a Path ending with
                # ".zip" (from _make_file_name).
                import zipfile

                zip_path = Path(save_fn)
                # inner asc filename is the zip name without the .zip suffix
                asc_name = zip_path.name[:-4]
                asc_temp = zip_path.parent / asc_name

                # write the ascii file
                need_zip = False
                with open(asc_temp, "w", encoding="utf-8") as fid:
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
                        self.logger.debug(f"END --> {time.ctime()}")
                        et = datetime.datetime.now()
                        write_time = et - st
                        self.logger.debug(
                            f"Writing took: {write_time.total_seconds()} seconds"
                        )
                        need_zip = True
                    else:
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

                # create zip and clean up (only if we need it)
                if need_zip:
                    try:
                        with zipfile.ZipFile(
                            zip_path, "w", compression=zipfile.ZIP_DEFLATED
                        ) as zf:
                            zf.write(asc_temp, arcname=asc_temp.name)
                    finally:
                        try:
                            asc_temp.unlink(missing_ok=True)
                        except Exception:
                            pass
                    return

            # default uncompressed write
            with open(save_fn, "w", encoding="utf-8") as fid:
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
                    self.logger.debug(f"END --> {time.ctime()}")
                    et = datetime.datetime.now()
                    write_time = et - st
                    self.logger.debug(
                        f"Writing took: {write_time.total_seconds()} seconds"
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
        self.logger.debug(f"END -->   {time.ctime()}")
        et = datetime.datetime.now()
        write_time = et - st
        self.logger.debug("Writing took: {write_time.total_seconds()} seconds")


def read_ascii(fn):
    """
    read USGS ASCII formatted file

    :param fn: DESCRIPTION
    :type fn: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    asc_obj = USGSascii(fn)
    asc_obj.read_ascii_file()

    return asc_obj.to_run_ts()
