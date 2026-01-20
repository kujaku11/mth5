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
def get_drives():
    """
    get a list of logical drives detected on the machine
    Note this only works for windows.
    Outputs:
    ----------
        **drives** : list of drives as letters
    :Example: ::
        >>> import mtpy.usgs.zen as zen
        >>> zen.get_drives()
    """
    drives = []
    bitmask = win32api.GetLogicalDrives()
    for letter in string.ascii_uppercase:
        if bitmask & 1:
            drives.append(letter)
        bitmask >>= 1
    return drives


# get the names of the drives which should correspond to channels
def get_drive_names():
    """
    get a list of drive names detected assuming the cards are names by box
    and channel.
    Outputs:
    ----------
        **drive_dict** : dictionary
                         keys are the drive letters and values are the
                         drive names
    :Example: ::
        >>> import mtpy.usgs.zen as zen
        >>> zen.get_drives_names()
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


def split_station(station):
    """
    split station name into name and number
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
    station,
    save_path=r"d:\Peacock\MTData",
    channel_dict={
        "1": "HX",
        "2": "HY",
        "3": "HZ",
        "4": "EX",
        "5": "EY",
        "6": "HZ",
    },
    copy_date=None,
    copy_type="all",
):
    """
    copy files from sd cards into a common folder (save_path)
    do not put an underscore in station, causes problems at the moment
    Arguments:
    -----------
        **station** : string
                      full name of station from which data is being saved
        **save_path** : string
                       full path to save data to
        **channel_dict** : dictionary
                           keys are the channel numbers as strings and the
                           values are the component that corresponds to that
                           channel, values are placed in upper case in the
                           code
        **copy_date** : YYYY-MM-DD
                        date to copy from depending on copy_type
        **copy_type** : [ 'all' | 'before' | 'after' | 'on' ]
                        * 'all' --> copy all files on the SD card
                        * 'before' --> copy files before and on this date
                        * 'after' --> copy files on and after this date
                        * 'on' --> copy files on this date only
    Outputs:
    -----------
        **fn_list** : list
                     list of filenames copied to save_path
    :Example: ::
        >>> import mtpy.usgs.zen as zen
        >>> fn_list = zen.copy_from_sd('mt01', save_path=r"/home/mt/survey_1")
    """
    s_name, s_int = split_station(station)
    drive_names = get_drive_names()
    save_path = Path(save_path).joinpath(station)

    if not save_path.exists():
        save_path.mkdir()
    if drive_names is None:
        logger.error("No drive names found. No files copied.")

        return [], save_path
    # make a datetime object from copy date
    if copy_date is not None:
        c_date = dateutil.parser.parse(copy_date)
    st_test = time.ctime()
    fn_list = []
    for key in list(drive_names.keys()):
        dr = Path(f"{key}:")
        logger.info(f"Reading from drive {key}.")

        for fn in list(set(list(dr.rglob("*.z3d")) + list(dr.rglob("*.Z3D")))):
            # test for copy date
            if copy_date is not None:
                file_date = datetime.datetime.fromtimestamp(fn.stat().st_mtime)
                if copy_type == "after":
                    if file_date < c_date:
                        continue
                elif copy_type == "before":
                    if file_date > c_date:
                        continue
                elif copy_type == "on":
                    if file_date.date() != c_date.date():
                        continue
            try:
                file_size = fn.stat().st_size
                if file_size >= 1600:
                    zt = Z3D(fn=fn)
                    zt.read_all_info()

                    if zt.metadata.station.find(s_int) >= 0:
                        channel = zt.metadata.ch_cmp.upper()
                        st = zt.schedule.Time.replace(":", "")
                        sd = zt.schedule.Date.replace("-", "")
                        sv_fn = (
                            f"{station}_{sd}_{st}_{int(zt.sample_rate)}_{channel}.Z3D"
                        )

                        new_fn = save_path.joinpath(sv_fn)
                        fn_list.append(new_fn)

                        shutil.copy(fn, new_fn)

                        logger.info(f"Copied {fn} to {new_fn}")
                        logger.info(f"File size is {file_size}")
                else:
                    logger.warning(f"Skipped {fn} because file to small {file_size}")
            except WindowsError:
                logger.warning(f"Faulty file at {fn}")
    et_test = time.ctime()
    logger.info(f"Started copying at: {st_test}")
    logger.info(f"Ended copying at:   {et_test}")

    return fn_list, save_path


# ==============================================================================
# delete files from sd cards
# ==============================================================================
def delete_files_from_sd(
    delete_date=None,
    delete_type=None,
    delete_folder=r"d:\Peacock\MTData\Deleted",
    verbose=True,
):
    """
    delete files from sd card, if delete_date is not None, anything on this
    date and before will be deleted.  Deletes just .Z3D files, leaves
    zenini.cfg
    Agruments:
    -----------
        **delete_date** : YYYY-MM-DD
                         date to delete files from
        **delete_type** : [ 'all' | 'before' | 'after' | 'on' ]
                          * 'all' --> delete all files on sd card
                          * 'before' --> delete files on and before delete_date
                          * 'after' --> delete files on and after delete_date
                          * 'on' --> delete files on delete_date
        **delete_folder** : string
                            full path to a folder where files will be moved to
                            just in case.  If None, files will be deleted
                            for ever.
    Returns:
    ---------
        **delete_fn_list** : list
                            list of deleted files.
     :Example: ::
        >>> import mtpy.usgs.zen as zen
        >>> # Delete all files before given date, forever.
        >>> zen.delete_files_from_sd(delete_date='2004/04/20',
                                     delete_type='before',
                                     delete_folder=None)
        >>> # Delete all files into a folder just in case
        >>> zen.delete_files_from_sd(delete_type='all',
                                     delete_folder=r"/home/mt/deleted_files")
    """

    drive_names = get_drive_names()
    if delete_folder is None:
        delete_path = Path().cwd()
    else:
        delete_path = Path(delete_folder)
        if not delete_path.exists():
            delete_path.mkdir()
    if drive_names is None:
        logger.error("No drives found.")
        raise IOError("No drives to copy from.")
    if delete_date is not None:
        delete_date = int(delete_date.replace("-", ""))
    delete_fn_list = []
    for key, value in drive_names.items():
        dr = Path(f"{key}:")
        for fn in dr.iterdir():
            if fn.suffix in [".Z3D", ".z3d"]:
                zt = Z3D(fn)
                zt.read_all_info()
                zt_date = int(zt.schedule.Date.replace("-", ""))
                # zt.get_info()
                if delete_type == "all" or delete_date is None:
                    if delete_folder is None:
                        fn.unlink()
                        delete_fn_list.append(fn)
                        logger.info(f"Deleted {fn}")
                    else:
                        shutil.move(fn, delete_path.joinpath(fn.name))
                        delete_fn_list.append(fn)
                        logger.info(f"Moved {fn} to {delete_path}")
                else:
                    if delete_type == "before":
                        if zt_date <= delete_date:
                            if delete_folder is None:
                                fn.unlink()
                                delete_fn_list.append(fn)
                                logger.info(f"Deleted {fn}")
                            else:
                                shutil.move(fn, delete_path.joinpath(fn.name))
                                delete_fn_list.append(fn)
                                logger.info(f"Moved {fn} to {delete_path}")
                    elif delete_type == "after":
                        if zt_date >= delete_date:
                            if delete_folder is None:
                                fn.unlink()
                                delete_fn_list.append(fn)
                                logger.info(f"Deleted {fn}")
                            else:
                                shutil.move(fn, delete_path.joinpath(fn.name))
                                delete_fn_list.append(fn)
                                logger.info(f"Moved {fn} to {delete_path}")
                    elif delete_type == "on":
                        if zt_date == delete_date:
                            if delete_folder is None:
                                fn.unlink()
                                delete_fn_list.append(fn)
                                logger.info(f"Deleted {fn}")
                            else:
                                shutil.move(fn, delete_path.joinpath(fn.name))
                                delete_fn_list.append(fn)
                                logger.info(f"Moved {fn} to {delete_path}")
    return delete_fn_list


# ==============================================================================
# read and write a zen schedule
# ==============================================================================
class ZenSchedule(object):
    """
    deals with reading, writing and copying schedule
    Creates a repeating schedule based on the master_schedule.  It will
    then change the first scheduling action to coincide with the master
    schedule, such that all deployed boxes will have the same schedule.
    :Example: ::
        >>> import mtpy.usgs.zen as zen
        >>> zs = zen.ZenSchedule()
        >>> zs.write_schedule('MT01', dt_offset='2013-06-23,04:00:00')
    ====================== ====================================================
    Attributes              Description
    ====================== ====================================================
    ch_cmp_dict            dictionary for channel components with keys being
                           the channel number and values being the channel
                           label
    ch_num_dict            dictionary for channel components whith keys
                           being channel label and values being channel number
    df_list                 sequential list of sampling rates to repeat in
                           schedule
    df_time_list            sequential list of time intervals to measure for
                           each corresponding sampling rate
    dt_format              date and time format. *default* is
                           YYY-MM-DD,hh:mm:ss
    dt_offset              start date and time of schedule in dt_format
    gain_dict              dictionary of gain values for channel number
    initial_dt             initial date, or dummy zero date for scheduling
    light_dict             dictionary of light color values for schedule
    master_schedule        the schedule that all data loggers should schedule
                           at.  Will taylor the schedule to match the master
                           schedule according to dt_offset
    meta_dict              dictionary for meta data
    meta_keys              keys for meta data dictionary
    sa_keys                keys for schedule actions
    sa_list                 list of schedule actions including time and df
    sr_dict                dictionary of sampling rate values
    verbose                [ True | False ] True to print information to
                           console
    ====================== ====================================================
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
        zen_start=None,
        df_list=None,
        df_time_list=None,
        repeat=8,
        gain=0,
        save_path=None,
        schedule_fn="zen_schedule.MTsch",
        version=4,
    ):
        """
        write a zen schedule file
        **Note**: for the older boxes use 'Zeus3Ini.cfg' for the savename
        Arguments:
        ----------
            **zen_start** : hh:mm:ss
                            start time you want the zen to start collecting
                            data.
                            if this is none then current time on computer is
                            used. **In UTC Time**
                            **Note**: this will shift the starting point to
                                      match the master schedule, so that all
                                      stations have the same schedule.
            **df_list** : list
                         list of sampling rates in Hz
            **df_time_list** : list
                              list of time intervals corresponding to df_list
                              in hh:mm:ss format
            **repeat** : int
                         number of time to repeat the cycle of df_list
            **gain** : int
                       gain on instrument, 2 raised to this number.
        Returns:
        --------
            * writes a schedule file to input into the ZenAcq Gui
        """

        if df_list is not None:
            self.df_list = df_list
        if df_time_list is not None:
            self.df_time_list = df_time_list
        if save_path is None:
            save_path = Path().cwd()
        else:
            save_path = Path(save_path)
        # make a master schedule first
        self.master_schedule = self.make_schedule(
            self.df_list, self.df_time_list, repeat=repeat * 3
        )
        # estimate the first off set time
        t_offset_dict = self.get_schedule_offset(zen_start, self.master_schedule)

        # make the schedule with the offset of the first schedule action
        self.sa_list = self.make_schedule(
            self.df_list,
            self.df_time_list,
            t1_dict=t_offset_dict,
            repeat=repeat,
        )

        # make a list of lines to write to a file for ZenAcq
        if version >= 4:
            zacq_list = ["$TX=0", "$Type=339"]
        elif version < 4:
            zacq_list = []
        for ii, ss in enumerate(self.sa_list[:-1]):
            t0 = self._convert_time_to_seconds(ss["time"])
            t1 = self._convert_time_to_seconds(self.sa_list[ii + 1]["time"])
            if ss["date"] != self.sa_list[ii + 1]["date"]:
                t1 += 24 * 3600
            # subtract 10 seconds for transition between schedule items.
            duration = t1 - t0 - self._resync_pause
            sr = int(self.sr_dict[str(ss["df"])])
            if version >= 4:
                zacq_list.append(f"$schline{ii+1} = {duration:.0f},{sr:.0f},1,0,0")
            elif version < 4:
                zacq_list.append(f"$schline{ii+1} = {duration:.0f},{sr:.0f},1")
        if version >= 4:
            zacq_list += [
                "$DayRepeat=0",
                "$RelativeOffsetSeconds=0",
                "$AutoSleep=0",
            ]
        fn = save_path.joinpath(schedule_fn)
        with open(fn, "w") as fid:
            fid.write("\n".join(zacq_list))
        print("Wrote schedule file to {0}".format(fn))
        print("+--------------------------------------+")
        print("|   SET ZEN START TIME TO: {0}    |".format(zen_start))
        print("+--------------------------------------+")

    def _convert_time_to_seconds(self, time_string):
        """
        convert a time string given as hh:mm:ss into seconds
        """
        t_list = [float(tt) for tt in time_string.split(":")]
        t_seconds = t_list[0] * 3600 + t_list[1] * 60 + t_list[2]

        return t_seconds
