"""
    This is a placeholder module for functions that are used in testing and development.
"""

from loguru import logger
from mth5.mth5 import MTH5
from mth5.utils.helpers import path_or_mth5_object
from typing import Union

import pathlib

GROUPBY_COLUMNS = ["survey", "station", "sample_rate"]


@path_or_mth5_object
def read_back_fcs(m: Union[MTH5, pathlib.Path, str], mode: str = "r") -> None:
    """
        Loops over stations in the channel summary of input (m) grouping by common sample_rate.
        Then loop over the runs in the corresponding FC Group.  Finally, within an fc_group,
        loop decimation levels and read data to xarray.  Log info about the shape of the xarray.

        This is a helper function for tests.  It was used as a sanity check while debugging the FC files, and
        also is a good example for how to access the data at each level for each channel.

        Development Notes:
        The Time axis of the FC array changes from decimation_level to decimation_level.
        The frequency axis will shape will depend on the window length that was used to perform STFT.
        This is currently storing all (positive frequency) fcs by default, but future versions can
        also have selected bands within an FC container.

        Parameters
        ----------
        m: Union[MTH5, pathlib.Path, str]
            Either a path to an mth5, or an MTH5 object that the FCs will be read back from.
        mode: str
            The mode to open the MTH5 file in. Defualts to (r)ead only.


    """
    channel_summary_df = m.channel_summary.to_dataframe()
    logger.debug(channel_summary_df)
    grouper = channel_summary_df.groupby(GROUPBY_COLUMNS)
    for (survey, station, sample_rate), group in grouper:
        logger.info(f"survey: {survey}, station: {station}, sample_rate {sample_rate}")
        station_obj = m.get_station(station, survey)
        fc_groups = station_obj.fourier_coefficients_group.groups_list
        logger.info(f"FC Groups: {fc_groups}")
        for run_id in fc_groups:
            fc_group = station_obj.fourier_coefficients_group.get_fc_group(run_id)
            dec_level_ids = fc_group.groups_list
            for dec_level_id in dec_level_ids:
                dec_level = fc_group.get_decimation_level(dec_level_id)
                xrds = dec_level.to_xarray(["hx", "hy"])
                msg = f"dec_level {dec_level_id}"
                msg = f"{msg} \n Time axis shape {xrds.time.data.shape}"
                msg = f"{msg} \n Freq axis shape {xrds.frequency.data.shape}"
                logger.debug(msg)

    return
