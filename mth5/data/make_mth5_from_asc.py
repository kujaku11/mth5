# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:03:21 2021

@author: jpeacock

This module is concerned with creating mth5 files from the synthetic test data
 that originally came from EMTF -- test1.asc and test2.asc.  Each ascii file
 represents five channels of data sampled at 1Hz at a synthetic station.

TODO: Separate the handling of legacy EMTF data files, such as
 reading into a dataframe from oddly delimited data, as well as flipping polarities of
 the electric channels (possibly due to a baked in sign convention error in the legacy
 data), so that a simple dataframe can be passed.  That will make the methods here more
  easily generalize to work with other dataframes.  That would be useful in future when
  we creating synthetic data at arbitrary sample rate.

Development Notes:
 Mirroring the original ascii files are:
 data/test1.h5
 data/test2.h5
 data/test12rr.h5

 Also created are some files with the same data but other channel_nomenclature schemes:
 data/test12rr_LEMI34.h5
 data/test1_LEMI12.h5

 - 20231103: Added an 8Hz up-sampled version of test1.  No spectral content was added
 so the band between the old and new Nyquist frequencies is bogus.



"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
import pathlib
import scipy.signal as ssig

from loguru import logger
from mth5.data.paths import SyntheticTestPaths
from mth5.data.station_config import make_filters
from mth5.data.station_config import make_station_01
from mth5.data.station_config import make_station_02
from mth5.data.station_config import make_station_03
from mth5.data.station_config import make_station_04
from mth5.data.station_config import SyntheticRun
from mth5.data.station_config import SyntheticStation
from mth5.mth5 import MTH5
from mth5.timeseries import ChannelTS, RunTS
from mth5.utils.helpers import add_filters
from mt_metadata.transfer_functions.processing.aurora import (
    ChannelNomenclature,
)
from mt_metadata.transfer_functions.processing.aurora.channel_nomenclature import SupportedNomenclature

from mt_metadata.timeseries import Electric
from mt_metadata.timeseries import Magnetic
from mt_metadata.timeseries import Survey

from typing import List, Literal, Optional, Union


# =============================================================================
np.random.seed(0)

synthetic_test_paths = SyntheticTestPaths()
MTH5_PATH = synthetic_test_paths.mth5_path


def create_run_ts_from_synthetic_run(
    run: SyntheticRun,
    df: pd.DataFrame,
    channel_nomenclature: SupportedNomenclature = "default"
):
    """
    Loop over channels of synthetic data in df and make ChannelTS objects.

    :param run: One-off data structure with information mth5 needs to initialize. Specifically sample_rate, filters.
    :type run: mth5.data.station_config.SyntheticRun
    :param df: time series data in columns labelled from ["ex", "ey", "hx", "hy", "hz"]
    :type df: pandas.DataFrame
    :param channel_nomenclature : Keyword corresponding to channel nomenclature mapping
    in CHANNEL_MAPS variable from channel_nomenclature.py module in mt_metadata.
    Supported values include ['default', 'lemi12', 'lemi34', 'phoenix123']
    :type channel_nomenclature : string

    :return runts: MTH5 run time series object, data and metadata bound into one.
    :rtype runts: RunTS

    """

    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature
    ch_list = []
    for i_col, col in enumerate(df.columns):

        data = df[col].values
        if col in channel_nomenclature_obj.ex_ey:
            channel_metadata = Electric()
            channel_metadata.units = "millivolts per kilometer"
        elif col in channel_nomenclature_obj.hx_hy_hz:
            channel_metadata = Magnetic()
            channel_metadata.units = "nanotesla"

        channel_metadata.component = col
        channel_metadata.channel_number = i_col  # not required
        channel_metadata.sample_rate = run.run_metadata.sample_rate
        channel_metadata.time_period.start = run.run_metadata.time_period.start
        chts = ChannelTS(
            channel_type=channel_metadata.type,  # "electric" or "magnetic"
            data=data,
            channel_metadata=channel_metadata.to_dict(),
        )

        # Set dipole properties
        # (Not sure how to pass this in channel_metadata when intializing)
        if col in channel_nomenclature_obj.ex_ey:
            chts.channel_metadata.dipole_length = 50
            if col == channel_nomenclature_obj.ey:
                chts.channel_metadata.measurement_azimuth = 90.0

        # Set filters
        chts.channel_metadata.filter.name = run.filters[col]
        chts.channel_metadata.filter.applied = len(run.filters[col]) * [
            True,
        ]

        ch_list.append(chts)

    # make a RunTS object
    runts = RunTS(array_list=ch_list, run_metadata=run.run_metadata)

    # add in metadata
    # runts.run_metadata.id = run.run_metadata.id
    return runts


def get_time_series_dataframe(
    run: SyntheticRun,
    source_folder: Union[pathlib.Path, str],
    add_nan_values: Optional[bool] = False
) -> pd.DataFrame:
    """
    Returns time series data in a dataframe with columns named for EM field component.

    Up-samples data to run.sample_rate, which is treated as in integer.
    Only tested for 8, to make 8Hz data for testing.  If run.sample_rate is default (1.0)
    then no up-sampling takes place.

    TODO: Move Upsampling, noise, and nan addition out of this method.

    :type run: mth5.data.station_config.SyntheticRun
    :param run: Information needed to define/create the run
    :type source_folder: Optional[Union[pathlib.Path, str]]
    :param source_folder: Where to load the ascii time series from.  This overwrites any
    previous value that may have been stored in the SyntheticRun
    :type add_nan_values: bool
    :param add_nan_values: If True, add some NaN, if False, do not add Nan.
    :rtype df: pandas.DataFrame
    :return df: The time series data for the synthetic run

    """
    # point to the ascii time series
    if source_folder:
        run.raw_data_path = source_folder.joinpath(run.raw_data_path.name)

    df = run._get_timeseries_dataframe()

    # upsample data if requested,
    if run.run_metadata.sample_rate != 1.0:
        df_orig = df.copy(deep=True)
        new_data_dict = {}
        for i_ch, ch in enumerate(run.channels):
            data = df_orig[ch].to_numpy()
            new_data_dict[ch] = ssig.resample(
                data, int(run.run_metadata.sample_rate) * len(df_orig)
            )
        df = pd.DataFrame(data=new_data_dict)

    # add noise
    for col in run.channels:
        if run.noise_scalars[col]:
            df[col] += run.noise_scalars[col] * np.random.randn(len(df))

    # add nan
    if add_nan_values:
        for col in run.channels:
            for [ndx, num_nan] in run.nan_indices[col]:
                df.loc[ndx: ndx + num_nan, col] = np.nan
    return df


def create_mth5_synthetic_file(
    station_cfgs: List[SyntheticStation],
    mth5_name: Union[pathlib.Path, str],
    target_folder: Optional[Union[pathlib.Path, str]] = "",
    source_folder: Union[pathlib.Path, str] = "",
    plot: bool = False,
    add_nan_values: bool = False,
    file_version: Literal["0.1.0", "0.2.0"] = "0.1.0",
    channel_nomenclature: SupportedNomenclature = "default",
    force_make_mth5: bool = True,
    survey_metadata: Optional[Survey] = None,
):
    """
    Creates an MTH5 from synthetic data.

    Development Notes:
     20250203: This function could be made more general, so that it operates on dataframes and legacy emtf ascii files.

    :param station_cfgs: Iterable of objects of type SyntheticStation. These are one-off
    data structure used to hold information mth5 needs to initialize, specifically
    sample_rate, filters, etc.
    :type station_cfgs: List[SyntheticStation]
    :param mth5_name: Where the mth5 will be stored.  This is generated by the station_config,
    but may change in this method based on add_nan_values or channel_nomenclature
    :type mth5_name: Union[pathlib.Path, str]
    :param target_folder: Where the mth5 file will be stored
    :type target_folder: Optional[Union[pathlib.Path, str]]
    :param source_folder:  Where the ascii source data are stored
    :type source_folder: Optional[Union[pathlib.Path, str]] = "",
    :param plot: Set to false unless you want to look at a plot of the time series
    :type plot: bool
    :param add_nan_values: If true, some np.nan are sprinkled into the time series.  Intended to be used for tests.
    :type add_nan_values: bool
    :param file_version: One of the supported mth5 file versions.  This is the version of mth5 to create.
    :type file_version: Literal["0.1.0", "0.2.0"] = "0.1.0",
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable,
    for example ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type channel_nomenclature: SupportedNomenclature
    :param force_make_mth5: If set to true, the file will be made, even if it already exists.
    If false, and file already exists, skip the make job.
    :type force_make_mth5: bool
    :param survey_metadata: Option to provide survey metadata, otherwise it will be created.
    :type survey_metadata: Survey
    :return: The path to the stored h5 file.
    :rtype: mth5_path: pathlib.Path

    """

    # Handle path and file name conventions
    if not target_folder:
        msg = f"No target folder provided for making {mth5_name}"
        logger.warning(msg)
        msg = f"Setting target folder to {MTH5_PATH}"
        logger.info(msg)
        target_folder = MTH5_PATH

    try:
        target_folder.mkdir(exist_ok=True, parents=True)
    except OSError:
        msg = "MTH5 maybe installed on a read-only file system"
        msg = f"{msg}: try setting `target_folder` argument when calling create_mth5_synthetic_file"
        logger.error(msg)

    mth5_path = target_folder.joinpath(mth5_name)
    mth5_path = _update_mth5_path(
        mth5_path, add_nan_values, channel_nomenclature
    )

    # Only create file if needed
    if not force_make_mth5:
        if mth5_path.exists():
            return mth5_path

    # create survey metadata:
    if not survey_metadata:
        survey_id = "EMTF Synthetic"
        survey_metadata = Survey()
        survey_metadata.id = survey_id

    # open output h5
    with MTH5(file_version=file_version) as m:
        m.open_mth5(mth5_path, mode="w")
        _add_survey(m, survey_metadata)

        for station_cfg in station_cfgs:
            station_group = m.add_station(station_cfg.id, survey=survey_id)

            for run in station_cfg.runs:
                # run is object of type SyntheticRun
                df = get_time_series_dataframe(
                    run=run,
                    source_folder=source_folder,
                    add_nan_values=add_nan_values
                )

                # TODO: Add handling for noise, nan, and upsampling here
                #  (They don't belong in get_time_Series_dataframe()

                # cast to run_ts
                runts = create_run_ts_from_synthetic_run(
                    run, df, channel_nomenclature=channel_nomenclature
                )
                runts.station_metadata.id = station_cfg.id

                # plot the data
                if plot:
                    runts.plot()

                run_group = station_group.add_run(run.run_metadata.id)
                run_group.from_runts(runts)

        # add filters
        active_filters = make_filters(as_list=True)
        add_filters(m, active_filters, survey_id)

    return mth5_path


def create_test1_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test1".

    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type file_version: str
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[Union[str, pathlib.Path]]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[Union[str, pathlib.Path]]
    :param source_folder:  Where the ascii source data are stored
    :type force_make_mth5: bool
    :param force_make_mth5: If set to true, the file will be made, even if it already exists.
    If false, and file already exists, skip the make job.
    :rtype: pathlib.Path
    :return: the path to the mth5 file

    """
    station_01_params = make_station_01(
        channel_nomenclature=channel_nomenclature
    )
    mth5_name = station_01_params.mth5_name
    station_params = [
        station_01_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
        target_folder=target_folder,
        source_folder=source_folder,
        force_make_mth5=force_make_mth5,
    )
    return mth5_path


def create_test2_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test2".

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :type force_make_mth5: bool
    :param force_make_mth5: If set to true, the file will be made, even if it already exists.
    If false, and file already exists, skip the make job.
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_02_params = make_station_02(
        channel_nomenclature=channel_nomenclature
    )
    mth5_name = station_02_params.mth5_name
    station_params = [
        station_02_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        file_version=file_version,
        force_make_mth5=force_make_mth5,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test1_h5_with_nan(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test1" with some nan values.

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_01_params = make_station_01(
        channel_nomenclature=channel_nomenclature
    )
    mth5_name = station_01_params.mth5_name
    station_params = [
        station_01_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        plot=False,
        add_nan_values=True,
        file_version=file_version,
        force_make_mth5=force_make_mth5,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test12rr_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file with data from two stations station named "test1" and "test2".

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_01_params = make_station_01(
        channel_nomenclature=channel_nomenclature
    )
    station_02_params = make_station_02(
        channel_nomenclature=channel_nomenclature
    )
    station_params = [station_01_params, station_02_params]
    mth5_name = "test12rr.h5"
    mth5_path = create_mth5_synthetic_file(
        station_params,
        mth5_name,
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
        target_folder=target_folder,
        source_folder=source_folder,
        force_make_mth5=force_make_mth5,
    )
    mth5_path = pathlib.Path(mth5_path)
    return mth5_path


def create_test3_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test3".
    This example has several runs and can be used to test looping over runs.

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :type force_make_mth5: bool
    :param force_make_mth5: If set to true, the file will be made, even if it already exists.
    If false, and file already exists, skip the make job.
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_03_params = make_station_03(
        channel_nomenclature=channel_nomenclature
    )
    station_params = [
        station_03_params,
    ]
    mth5_path = create_mth5_synthetic_file(
        station_params,
        station_params[0].mth5_name,
        file_version=file_version,
        force_make_mth5=force_make_mth5,
        target_folder=target_folder,
        source_folder=source_folder,
    )
    return mth5_path


def create_test4_h5(
    file_version: Optional[str] = "0.1.0",
    channel_nomenclature: Optional[str] = "default",
    target_folder: Optional[Union[str, pathlib.Path]] = MTH5_PATH,
    source_folder: Optional[Union[str, pathlib.Path]] = "",
    force_make_mth5: Optional[bool] = True,
) -> pathlib.Path:
    """
    Creates an MTH5 file for a single station named "test1", data are up-sampled to 8Hz from
    original 1 Hz.

    Note: Because the 8Hz data are derived from the 1Hz, only frequencies below 0.5Hz
    will have valid TFs that yield the apparent resistivity of the synthetic data (100 Ohm-m).

    :type file_version: str
    :param file_version: One of ["0.1.0", "0.2.0"], corresponding to the version of mth5 to create
    :type channel_nomenclature: Optional[str]
    :param channel_nomenclature: Keyword corresponding to channel nomenclature mapping in CHANNEL_MAPS variable
    from channel_nomenclature.py module in mt_metadata. Supported values are ['default', 'lemi12', 'lemi34', 'phoenix123']
    A full list is in mt_metadata/transfer_functions/processing/aurora/standards/channel_nomenclatures.json
    :type target_folder: Optional[str, pathlib.Path]
    :param target_folder: Where the mth5 file will be stored
    :type source_folder: Optional[str, pathlib.Path]
    :param source_folder:  Where the ascii source data are stored
    :rtype: pathlib.Path
    :return: the path to the mth5 file
    """
    station_04_params = make_station_04(
        channel_nomenclature=channel_nomenclature
    )
    mth5_path = create_mth5_synthetic_file(
        [
            station_04_params,
        ],
        station_04_params.mth5_name,
        plot=False,
        file_version=file_version,
        channel_nomenclature=channel_nomenclature,
        target_folder=target_folder,
        source_folder=source_folder,
        force_make_mth5=force_make_mth5,
    )
    return mth5_path


def _add_survey(m: MTH5, survey_metadata: Survey) -> None:
    """
    :type m: mth5.mth5.MTH5
    :param m: The mth5 object to get/set survey_id with
    :type survey_metadata: mt_metadata.timeseries.Survey
    :param survey_metadata: The survey metadata in mt_metadata container

    """
    if m.file_version == "0.1.0":
        # "no need to pass survey id in v 0.1.0 -- just the metadata"
        m.survey_group.update_metadata(survey_metadata.to_dict())
    elif m.file_version == "0.2.0":
        m.add_survey(survey_metadata.id, survey_metadata)
    else:
        msg = f"unexpected MTH5 file_version = {m.file_version}"
        raise NotImplementedError(msg)
    return


def _update_mth5_path(
    mth5_path: pathlib.Path,
    add_nan_values: bool,
    channel_nomenclature: str
) -> pathlib.Path:
    """ Modify the name of output h5 file based on wheter or not nan-data are included
     as well as channel_nomenclature if not default. """
    path_str = mth5_path.__str__()
    if add_nan_values:
        path_str = path_str.replace(".h5", "_nan.h5")
    if channel_nomenclature != "default":
        path_str = path_str.replace(".h5", f"_{channel_nomenclature}.h5")
    return pathlib.Path(path_str)


def main(file_version="0.1.0"):
    """Allow the module to be called from the command line"""
    create_test1_h5(file_version=file_version)
    create_test1_h5_with_nan(file_version=file_version)
    create_test2_h5(file_version=file_version)
    create_test12rr_h5(
        file_version=file_version, channel_nomenclature="lemi12"
    )
    create_test3_h5(file_version=file_version)
    create_test4_h5(file_version=file_version)


if __name__ == "__main__":
    main(file_version="0.2.0")
    main(file_version="0.1.0")
