from dataclasses import dataclass
from loguru import logger
from typing import Tuple  #, Union
import pandas as pd

@dataclass
class FCRunChunk():
    """
    Just a container to formalize the elements of the Tuple needed.  This may become formalizec in mt_metadata,
    for now just use a dataclass.
    """
    survey_id: str = "none"
    station_id: str = ""
    run_id: str = ""
    decimation_level_id: str = "0"
    start: str = ""
    end: str = ""
    channels: Tuple[str] = ()

    @property
    def start_timestamp(self):
        return pd.Timestamp(self.start)

    @property
    def end_timestamp(self):
        return pd.Timestamp(self.end)



def make_multistation_spectrogram(m, fc_run_chunks):
    """
    see notes in mth5 issue #209.  Takes a list of FCRunChunks and returns the largest contiguous
    block of multichannel FC data available.

    |----------Station 1 ------------|
            |----------Station 2 ------------|
    |--------------------Station 3 ----------------------|


            |-------RETURNED------|

    Handle additional runs in a separate call to this function and then concatenate time series afterwards.

    Input must specify N (station-run-start-end-channel_list) tuples.
    If channel_list is not provided, get all channels.
    If start-end are not provided, read the whole run -- warn if runs are not all synchronous, and
    truncate all to max(starts), min(ends) after the start and end times are sorted out.

    Station IDs must be unique.


    Parameters
    ----------
    m: mth5.mth5.MTH5
        The mth5 object to get the FCs from.
    fc_run_chunks: iterable
        Each element of this describes the run chunk to load.

    Returns
    -------

    """
    for i_fcrc, fcrc in enumerate(fc_run_chunks):
        station_obj = m.get_station(fcrc.station_id, fcrc.survey_id)
        station_fc_group = station_obj.fourier_coefficients_group
        logger.info(f"Available FC Groups for station {fcrc.station_id}: {station_fc_group.groups_list}")
        run_fc_group = station_obj.fourier_coefficients_group.get_fc_group(fcrc.run_id)
        # print(run_fc_group)
        fc_dec_level = run_fc_group.get_decimation_level(fcrc.decimation_level_id)
        if fcrc.channels:
            channels = list(fcrc.channels)
        else:
            channels = None

        fc_dec_level_xrds = fc_dec_level.to_xarray(channels=channels)
        # could create name mapper dict from run_fc_group.channel_summary here if we wanted to.
        if fcrc.start:
            # TODO: Push this into the to_xarray command so we only access what we want
            cond = fc_dec_level_xrds.time >= fcrc.start_timestamp
            msg = (
                f"trimming  {sum(~cond.data)} samples to {fcrc.start} "
            )
            logger.info(msg)
            fc_dec_level_xrds = fc_dec_level_xrds.where(cond)
            fc_dec_level_xrds = fc_dec_level_xrds.dropna(dim="time")

        if fcrc.end:
            # TODO: Push this into the to_xarray command so we only access what we want
            cond = fc_dec_level_xrds.time <= fcrc.end_timestamp
            msg = (
                f"trimming  {sum(~cond.data)} samples to {fcrc.end} "
            )
            logger.info(msg)
            fc_dec_level_xrds = fc_dec_level_xrds.where(cond)
            fc_dec_level_xrds = fc_dec_level_xrds.dropna(dim="time")


        name_dict = {f"{x}": f"{fcrc.station_id}_{x}" for x in fc_dec_level_xrds.data_vars}

        if i_fcrc == 0:
            output = fc_dec_level_xrds.rename_vars(name_dict=name_dict)
        else:
            fc_dec_level_xrds = fc_dec_level_xrds.rename_vars(name_dict=name_dict)
            output = output.merge(fc_dec_level_xrds)

    return output
