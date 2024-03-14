# =============================================================================
# Imports
# =============================================================================
import functools
import pathlib

from loguru import logger

from mth5.helpers import close_open_files
from mth5.mth5 import MTH5

# =============================================================================


def path_or_mth5_object(func):
    """
    Decorator to allow functions to be written as if an mth5_object was passed as first argument.

    TODO: add support for file_version in kwargs

    Parameters
    ----------
    func: function
        A function that takes as first argument an mth5.mth5.MTH5 object

    Returns
    -------

    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        def call_function(func, *args, **kwargs):
            if isinstance(func, staticmethod):
                callable_func = func.__get__(None, object)
                result = callable_func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result

        if isinstance(args[0], (pathlib.Path, str)):
            h5_path = args[0]
            mode = kwargs.get("mode", "a")
            #with MTH5().open_mth5(h5_path, mode=mode) as m:
            with MTH5() as m:
                m.open_mth5(h5_path, mode=mode)
                new_args = [x for x in args]
                new_args[0] = m
                new_args = tuple(new_args)
                result = call_function(func, *new_args, **kwargs)

        elif isinstance(args[0], MTH5):
            result = call_function(func, *args, **kwargs)
        else:
            msg = f"expected h5, got {type(args[0])}"
            logger.error(msg)
            raise TypeError(msg)

        return result

    return wrapper_decorator

@path_or_mth5_object
def get_version(m):
    return m.file_version

@path_or_mth5_object
def get_channel_summary(m, show=True):
    """
    :param mth5_path:
    :return:
    """
    logger.info(f"{m.filename} channel summary")
    df = m.channel_summary.to_dataframe()
    if len(df) <= 1:
        logger.warning("channel summary smaller than expected -- re-summarizing")
        m.channel_summary.summarize()
        df = m.channel_summary.to_dataframe()
    if show:
        logger.info(f"{df}")
    return df


@path_or_mth5_object
def add_filters(m, filters_list, survey_id=""):
    """

    Parameters
    ----------
    active_filters: list of filters
    m: mth5.mth5.MTH5
    survey_id: string

    Returns
    -------

    """
    if m.file_version == "0.1.0":
        fg = m.filters_group
    else:
        # m.file_version == "0.2.0":
        survey = m.get_survey(survey_id)
        fg = survey.filters_group

    for filt3r in filters_list:
        if filt3r.name not in fg.filter_dict.keys():
            fg.add_filter(filt3r)
    return


def initialize_mth5(h5_path, mode="a", file_version="0.1.0"):
    """
    mth5 initializer for the case of writting files.


    :param h5_path: path to file
    :type h5_path: string or pathlib.Path
    :param mode: how to open the file, options are

        - "r": read
        - "w": write
        - "a": append

    :type mode: string
    :return: mth5 object
    :rtype: :class:`mth5.MTH5`


    """
    h5_path = pathlib.Path(h5_path)
    if mode == "w":
        if h5_path.exists():
            msg = f"File {h5_path} exists, removing from file system."
            msg = f"{msg}\n closing all open h5 files before removal"
            logger.warning(f"{msg}")
            close_open_files()
            h5_path.unlink()
    mth5_obj = MTH5(file_version=file_version)
    mth5_obj.open_mth5(str(h5_path), mode=mode)

    return mth5_obj


def read_back_data(
    mth5_path,
    station_id,
    run_id,
    survey=None,
    close_mth5=True,
    return_objects=[],
):
    """
    TODO: add path_or_mth5_decorater to this function
    Testing helper function, used to confirm that the h5 file can be accessed
    and that the data size is as expected.

    :param mth5_path: the full path the the mth5 that this method is going to
     try to read
    :type mth5_path: pathlib.Path or string
    :param station_id: the label for the station, e.g. "PKD"
    :type station_id: string
    :param run_id: The label for the run to read.  e.g. "001"
    :type run_id: string
    :param survey: The label for the survey associated with the run to read.
    :type survey: string
    :param close_mth5: Whether or not to close the mth5 object after reading
    :type close_mth5: bool
    :param return_objects: List of strings.  Specifies what, if anything to return.
    Allowed values: ["run", "run_ts"]
    :type return_objects: List of strings.
    :return: run object
    :rtype: :class:`mth5.groups.RunGroup`
    :return: run time series
    :rtype: :class:`mth5.timeseries.RunTS`

    """
    processing_config = {}
    processing_config["mth5_path"] = str(mth5_path)
    processing_config["local_station_id"] = station_id
    config = processing_config
    m = initialize_mth5(config["mth5_path"], mode="r")
    local_run_obj = m.get_run(
        config["local_station_id"], run_id, survey=survey
    )
    local_run_ts = local_run_obj.to_runts()
    data_array = local_run_ts.dataset.to_array()
    logger.info(f"data shape = {data_array.shape}")

    return_dict = {}
    if "run" in return_objects:
        return_dict["run"] = local_run_obj
    if "run_ts" in return_objects:
        return_dict["run_ts"] = local_run_ts
    if close_mth5:
        m.close_mth5()
    else:
        return_dict["mth5_obj"] = m
    return return_dict


def get_compare_dict(input_dict):
    """
    Helper function for removing 2 added attributes to metadata

     - hdf5_reference
     - mth5_type

    :param input_dict: DESCRIPTION
    :type input_dict: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    for key in ["hdf5_reference", "mth5_type"]:
        try:
            input_dict.pop(key)
        except KeyError:
            pass

    return input_dict
