# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:40:28 2023

@author: jpeacock
"""
import datetime
import shutil
import string
import time

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import dateutil.parser
import numpy as np
from loguru import logger

from mth5.io.zen import Z3D


try:
    import win32api
except ImportError:
    print("WARNING: Cannot find win32api, will not be able to detect" " drive names")
# =============================================================================


# get the external drives for SD cards
def get_drives() -> list[str]:
    """
    Get a list of logical drives detected on the machine.

    Note: This only works for Windows.

    Returns
    -------
    list of str
        List of drives as letters.

    Examples
    --------
    >>> get_drives()
    ['C', 'D', 'E']
    """
    drives = []
    bitmask = win32api.GetLogicalDrives()
    for letter in string.ascii_uppercase:
        if bitmask & 1:
            drives.append(letter)
        bitmask >>= 1
    return drives


# get the names of the drives which should correspond to channels
def get_drive_names() -> dict[str, str] | None:
    """
    Get a list of drive names detected assuming the cards are named by box and channel.

    Returns
    -------
    dict of str to str or None
        Keys are the drive letters and values are the drive names. Returns None if no drives are found.

    Examples
    --------
    >>> get_drive_names()
    {'D': 'CH1', 'E': 'CH2'}
    """
    drives = get_drives()

    drive_dict = {}
    for drive in drives:
        try:
            drive_name = win32api.GetVolumeInformation(drive + ":\\")[0]
            if drive_name.find("CH") >= 0:
                drive_dict[drive] = drive_name
        except:
            pass
    if not bool(drive_dict):
        return None
    return drive_dict


def split_station(station: str) -> tuple[str, str]:
    """
    Split station name into name and number.

    Parameters
    ----------
    station : str
        Full station name.

    Returns
    -------
    tuple of str
        Tuple containing the station name and number.

    Examples
    --------
    >>> split_station('MT01')
    ('MT', '01')
    """

    for ii, ss in enumerate(station):
        try:
            int(ss)
            find = ii
            break
        except ValueError:
            continue
    name = station[0:find]
    number = station[find:]

    return (name, number)


def copy_from_sd(
    station: str,
    save_path: str | Path = Path(r"d:\\Peacock\\MTData"),
    channel_dict: dict[str, str] = {
        "1": "HX",
        "2": "HY",
        "3": "HZ",
        "4": "EX",
        "5": "EY",
        "6": "HZ",
    },
    copy_date: str | None = None,
    copy_type: str = "all",
) -> tuple[list[Path], Path]:
    """
    Copy files from SD cards into a common folder (save_path).

    Parameters
    ----------
    station : str
        Full name of station from which data is being saved.
    save_path : str or Path, optional
        Full path to save data to, by default 'd:\\Peacock\\MTData'.
    channel_dict : dict of str to str, optional
        Keys are the channel numbers as strings and the values are the component that corresponds to that channel. Values are placed in upper case in the code.
    copy_date : str, optional
        Date to copy from depending on copy_type, in 'YYYY-MM-DD' format.
    copy_type : {'all', 'before', 'after', 'on'}, optional
        Type of copy operation:
        - 'all': Copy all files on the SD card.
        - 'before': Copy files before and on this date.
        - 'after': Copy files on and after this date.
        - 'on': Copy files on this date only.

    Returns
    -------
    tuple of list of Path and Path
        List of filenames copied to save_path and the save_path itself.

    Examples
    --------
    >>> copy_from_sd('MT01', save_path=r"/home/mt/survey_1")
    ([Path('/home/mt/survey_1/MT01_2026-01-27_120000_256_HX.Z3D')], Path('/home/mt/survey_1'))
    """
    save_path = Path(save_path).joinpath(station)

    if not save_path.exists():
        save_path.mkdir(parents=True)

    fn_list = []
    drive_names = get_drive_names()
    if drive_names is None:
        logger.error("No drive names found. No files copied.")
        return [], save_path

    if copy_date is not None:
        c_date = dateutil.parser.parse(copy_date)

    s_name, s_int = split_station(station)

    for key, drive_name in drive_names.items():
        dr = Path(f"{key}:")
        logger.info(f"Reading from drive {key}.")

        for fn in dr.rglob("*.z3d"):
            if copy_date is not None:
                file_date = datetime.datetime.fromtimestamp(fn.stat().st_mtime)
                if (
                    (copy_type == "after" and file_date < c_date)
                    or (copy_type == "before" and file_date > c_date)
                    or (copy_type == "on" and file_date.date() != c_date.date())
                ):
                    continue

            try:
                file_size = fn.stat().st_size
                if file_size >= 1600:
                    zt = Z3D(fn=fn)
                    zt.read_all_info()

                    if (
                        zt.metadata
                        and zt.metadata.station
                        and s_int in zt.metadata.station
                    ):
                        channel = (
                            zt.metadata.ch_cmp.upper()
                            if zt.metadata.ch_cmp
                            else "UNKNOWN"
                        )
                        st = (
                            zt.schedule.Time.replace(":", "")
                            if zt.schedule and zt.schedule.Time
                            else "000000"
                        )
                        sd = (
                            zt.schedule.Date.replace("-", "")
                            if zt.schedule and zt.schedule.Date
                            else "000000"
                        )

                        sv_fn = f"{station}_{sd}_{st}_{int(zt.sample_rate or 0)}_{channel}.Z3D"
                        new_fn = save_path / sv_fn
                        fn_list.append(new_fn)

                        shutil.copy(fn, new_fn)
                        logger.info(f"Copied {fn} to {new_fn}")
                        logger.info(f"File size is {file_size}")
                else:
                    logger.warning(f"Skipped {fn} because file too small {file_size}")
            except Exception as e:
                logger.warning(f"Error processing file {fn}: {e}")

    return fn_list, save_path


# ==============================================================================
# delete files from sd cards
# ==============================================================================
def delete_files_from_sd(
    delete_date: str | None = None,
    delete_type: str | None = None,
    delete_folder: str | Path = Path().cwd(),
    verbose: bool = True,
) -> list[Path]:
    """
    Delete files from SD card. If delete_date is not None, anything on this date and before will be deleted.

    Parameters
    ----------
    delete_date : str, optional
        Date to delete files from, in 'YYYY-MM-DD' format.
    delete_type : {'all', 'before', 'after', 'on'}, optional
        Type of delete operation:
        - 'all': Delete all files on SD card.
        - 'before': Delete files on and before delete_date.
        - 'after': Delete files on and after delete_date.
        - 'on': Delete files on delete_date.
    delete_folder : str or Path, optional
        Full path to a folder where files will be moved to just in case. If None, files will be deleted permanently.
    verbose : bool, optional
        If True, print detailed logs, by default True.

    Returns
    -------
    list of Path
        List of deleted files.

    Examples
    --------
    >>> delete_files_from_sd(delete_date='2026-01-27', delete_type='before', delete_folder=None)
    [Path('D:/file1.Z3D'), Path('D:/file2.Z3D')]
    """
    delete_path = Path(delete_folder) if delete_folder else Path.cwd()
    if not delete_path.exists():
        delete_path.mkdir(parents=True)

    delete_fn_list = []
    drive_names = get_drive_names()
    if drive_names is None:
        logger.error("No drives found.")
        raise OSError("No drives found.")

    for key, drive_name in drive_names.items():
        dr = Path(f"{key}:")
        for fn in dr.iterdir():
            if fn.suffix.lower() == ".z3d":
                zt = Z3D(fn)
                zt.read_all_info()
                zt_date = (
                    int(zt.schedule.Date.replace("-", ""))
                    if zt.schedule and zt.schedule.Date
                    else 0
                )

                if delete_type == "all" or delete_date is None:
                    target = delete_path / fn.name if delete_folder else None
                    if target:
                        shutil.move(fn, target)
                    else:
                        fn.unlink()
                    delete_fn_list.append(fn)
                elif delete_type == "before" and zt_date <= int(
                    delete_date.replace("-", "")
                ):
                    target = delete_path / fn.name if delete_folder else None
                    if target:
                        shutil.move(fn, target)
                    else:
                        fn.unlink()
                    delete_fn_list.append(fn)
                elif delete_type == "after" and zt_date >= int(
                    delete_date.replace("-", "")
                ):
                    target = delete_path / fn.name if delete_folder else None
                    if target:
                        shutil.move(fn, target)
                    else:
                        fn.unlink()
                    delete_fn_list.append(fn)
                elif delete_type == "on" and zt_date == int(
                    delete_date.replace("-", "")
                ):
                    target = delete_path / fn.name if delete_folder else None
                    if target:
                        shutil.move(fn, target)
                    else:
                        fn.unlink()
                    delete_fn_list.append(fn)

    return delete_fn_list


# ==============================================================================
# read and write a zen schedule
# ==============================================================================
class ZenSchedule:
    """
    Deals with reading, writing, and copying schedules.

    Creates a repeating schedule based on the master_schedule. It will then change the first scheduling action to coincide with the master schedule, such that all deployed boxes will have the same schedule.

    Attributes
    ----------
    verbose : bool
        If True, print detailed logs.
    sr_dict : dict of str to str
        Dictionary of sampling rate values.
    sa_list : list of dict
        List of schedule actions including time and df.
    ch_cmp_dict : dict of str to str
        Dictionary for channel components with keys being the channel number and values being the channel label.
    ch_num_dict : dict of str to str
        Dictionary for channel components with keys being the channel label and values being the channel number.
    dt_format : str
        Date and time format, default is 'YYYY-MM-DD,hh:mm:ss'.
    initial_dt : str
        Initial date, or dummy zero date for scheduling.
    dt_offset : str
        Start date and time of schedule in dt_format.
    df_list : tuple of int
        Sequential list of sampling rates to repeat in schedule.
    df_time_list : tuple of str
        Sequential list of time intervals to measure for each corresponding sampling rate.
    master_schedule : list of dict
        The schedule that all data loggers should schedule at. Will tailor the schedule to match the master schedule according to dt_offset.
    """

    def __init__(self):
        self.verbose = True
        self.sr_dict = {
            "256": "0",
            "512": "1",
            "1024": "2",
            "2048": "3",
            "4096": "4",
        }
        self.sa_list = []
        self.ch_cmp_dict = {
            "1": "hx",
            "2": "hy",
            "3": "hz",
            "4": "ex",
            "5": "ey",
            "6": "hz",
        }
        self.ch_num_dict = dict(
            [(self.ch_cmp_dict[key], key) for key in self.ch_cmp_dict]
        )

        self.dt_format = "%Y-%m-%d,%H:%M:%S"
        self.initial_dt = "2000-01-01,00:00:00"
        self.dt_offset = time.strftime(self.dt_format, time.gmtime())
        self.df_list = (4096, 256)
        self.df_time_list = ("00:10:00", "07:50:00")
        self.master_schedule = self.make_schedule(
            self.df_list, self.df_time_list, repeat=16
        )
        self._resync_pause = 20

    # ==================================================
    def add_time(
        self, date_time, add_minutes=0, add_seconds=0, add_hours=0, add_days=0
    ):
        """
        add time to a time string
        assuming date_time is in the format  YYYY-MM-DD,HH:MM:SS
        """

        fulldate = datetime.datetime.strptime(date_time, self.dt_format)

        fulldate = fulldate + datetime.timedelta(
            days=add_days,
            hours=add_hours,
            minutes=add_minutes,
            seconds=add_seconds,
        )
        return fulldate

    # ==================================================
    def make_schedule(self, df_list, df_length_list, repeat=5, t1_dict=None):
        """
        make a repeated schedule given list of sampling frequencies and
        duration for each.
        Arguments:
        -----------
            **df_list** : list
                         list of sampling frequencies in Hz, note needs to be
                         powers of 2 starting at 256
            **df_length_list** : list
                                list of durations in hh:mm:ss format
            **repeat** : int
                         number of times to repeat the sequence
            **t1_dict** : dictionary
                          dictionary returned from get_schedule_offset
        Returns:
        --------
            **time_list**: list of dictionaries with keys:
                            * 'dt' --> date and time of schedule event
                            * 'df' --> sampling rate for that event
        """

        df_list = np.array(df_list)
        df_length_list = np.array(df_length_list)
        ndf = len(df_list)

        if t1_dict is not None:
            time_list = [{"dt": self.initial_dt, "df": t1_dict["df"]}]

            kk = np.where(np.array(df_list) == t1_dict["df"])[0][0] - ndf + 1
            df_list = np.append(df_list[kk:], df_list[:kk])
            df_length_list = np.append(df_length_list[kk:], df_length_list[:kk])
            time_list.append(dict([("dt", t1_dict["dt"]), ("df", df_list[0])]))
            ii = 1
        else:
            time_list = [{"dt": self.initial_dt, "df": df_list[0]}]
            ii = 0
        for rr in range(1, repeat + 1):
            for df, df_length, jj in zip(df_list, df_length_list, range(ndf)):
                dtime = time.strptime(df_length, "%H:%M:%S")
                ndt = self.add_time(
                    time_list[ii]["dt"],
                    add_hours=dtime.tm_hour,
                    add_minutes=dtime.tm_min,
                    add_seconds=dtime.tm_sec,
                )
                time_list.append(
                    {
                        "dt": ndt.strftime(self.dt_format),
                        "df": df_list[jj - ndf + 1],
                    }
                )
                ii += 1
        for nn, ns in enumerate(time_list):
            sdate, stime = ns["dt"].split(",")
            ns["date"] = sdate
            ns["time"] = stime
            ns["sr"] = self.sr_dict[str(ns["df"])]

        return time_list

    # ==================================================
    def get_schedule_offset(self, time_offset, schedule_time_list):
        """
        gets the offset in time from master schedule list and time_offset so
        that all schedules will record at the same time according to master
        schedule list schedule_time_list
        Attributes:
        -----------
            **time_offset** : hh:mm:ss
                              the time offset given to the zen reciever
            **schedule_time_list** : list
                                    list of actual schedule times returned
                                    from make_schedule
        Returns:
        --------
            **s1** : dictionary
                     dictionary with keys:
                         * 'dt' --> date and time of offset from next schedule
                                    event from schedule_time_list
                         * 'df' --> sampling rate of that event
        """

        dt_offset = "{0},{1}".format("2000-01-01", time_offset)
        t0 = time.mktime(time.strptime("2000-01-01,00:00:00", self.dt_format))

        for ii, tt in enumerate(schedule_time_list):
            ssec = time.mktime(time.strptime(tt["dt"], self.dt_format))
            osec = time.mktime(time.strptime(dt_offset, self.dt_format))

            if ssec > osec:
                sdiff = time.localtime(t0 + (ssec - osec))
                t1 = self.add_time(
                    "2000-01-01,00:00:00",
                    add_hours=sdiff.tm_hour,
                    add_minutes=sdiff.tm_min,
                    add_seconds=sdiff.tm_sec,
                )
                s1 = {
                    "dt": t1.strftime(self.dt_format),
                    "df": schedule_time_list[ii - 1]["df"],
                }
                return s1

    def write_schedule_for_gui(
        self,
        zen_start: str | None = None,
        df_list: list[int] | None = None,
        df_time_list: list[str] | None = None,
        repeat: int = 8,
        gain: int = 0,
        save_path: str | Path | None = None,
        schedule_fn: str = "zen_schedule.MTsch",
        version: int = 4,
    ) -> None:
        """
        Write a zen schedule file.

        Parameters
        ----------
        zen_start : str, optional
            Start time you want the zen to start collecting data, in UTC time. If None, current time is used.
        df_list : list of int, optional
            List of sampling rates in Hz.
        df_time_list : list of str, optional
            List of time intervals corresponding to df_list in hh:mm:ss format.
        repeat : int, optional
            Number of times to repeat the cycle of df_list, by default 8.
        gain : int, optional
            Gain on instrument, 2 raised to this number, by default 0.
        save_path : str or Path, optional
            Path to save the schedule file, by default current working directory.
        schedule_fn : str, optional
            Name of the schedule file, by default 'zen_schedule.MTsch'.
        version : int, optional
            Version of the schedule file format, by default 4.

        Returns
        -------
        None
        """
        if df_list is not None:
            self.df_list = df_list
        if df_time_list is not None:
            self.df_time_list = df_time_list
        if save_path is None:
            save_path = Path.cwd()
        else:
            save_path = Path(save_path)

        self.master_schedule = self.make_schedule(
            self.df_list, self.df_time_list, repeat=repeat * 3
        )

        t_offset_dict = self.get_schedule_offset(zen_start, self.master_schedule)

        self.sa_list = self.make_schedule(
            self.df_list,
            self.df_time_list,
            t1_dict=t_offset_dict,
            repeat=repeat,
        )

        zacq_list: list[str] = []
        if version >= 4:
            zacq_list = ["$TX=0", "$Type=339"]

        for ii, ss in enumerate(self.sa_list[:-1]):
            t0 = self._convert_time_to_seconds(ss["time"])
            t1 = self._convert_time_to_seconds(self.sa_list[ii + 1]["time"])
            if ss["date"] != self.sa_list[ii + 1]["date"]:
                t1 += 24 * 3600

            duration = t1 - t0 - self._resync_pause
            sr = int(
                self.sr_dict[str(ss["df"])] if str(ss["df"]) in self.sr_dict else 0
            )

            if version >= 4:
                zacq_list.append(f"$schline{ii+1} = {duration:.0f},{sr:.0f},1,0,0")
            else:
                zacq_list.append(f"$schline{ii+1} = {duration:.0f},{sr:.0f},1")

        if version >= 4:
            zacq_list.extend(
                [
                    "$DayRepeat=0",
                    "$RelativeOffsetSeconds=0",
                    "$AutoSleep=0",
                ]
            )

        fn = save_path / schedule_fn
        with open(fn, "w") as fid:
            fid.write("\n".join(zacq_list))

        print(f"Wrote schedule file to {fn}")
        print("+--------------------------------------+")
        print(f"|   SET ZEN START TIME TO: {zen_start}    |")
        print("+--------------------------------------+")

    def _convert_time_to_seconds(self, time_string):
        """
        convert a time string given as hh:mm:ss into seconds
        """
        t_list = [float(tt) for tt in time_string.split(":")]
        t_seconds = t_list[0] * 3600 + t_list[1] * 60 + t_list[2]

        return t_seconds
