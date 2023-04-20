# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:40:28 2023

@author: jpeacock
"""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import time
import datetime
import dateutil.parser
import string
import shutil

from mth5.utils.mth5_logger import setup_logger
from mth5.io.zen import Z3D

try:
    import win32api
except ImportError:
    print(
        "WARNING: Cannot find win32api, will not be able to detect"
        " drive names"
    )
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

    logger = setup_logger(
        "copy_z3d_from_sd",
        fn=save_path.joinpath("copy_from_sd.log"),
        level="debug",
    )

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
                        sv_fn = f"{station}_{sd}_{st}_{int(zt.sample_rate)}_{channel}.Z3D"

                        new_fn = save_path.joinpath(sv_fn)
                        fn_list.append(new_fn)

                        shutil.copy(fn, new_fn)

                        logger.info(f"Copied {fn} to {new_fn}")
                        logger.info(f"File size is {file_size}")

                else:
                    logger.warning(
                        f"Skipped {fn} because file to small {file_size}"
                    )
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

    logger = setup_logger(
        "delete_z3d_from_sd",
        fn=delete_path.joinpath("delete_from_sd.log"),
        level="debug",
    )
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
