from typing import Tuple, List, Union
from dataclasses import dataclass

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

    TODO: Update this to take [ { station1:station1run,
                                station2:station2run,
                                ...
                                stationN:stationNrun},
                                ...
                                {}
                                ]
    Parameters
    ----------
    m
    station_ids
    run_id
    decimation_level_id

    Returns
    -------

    """
    for i_fcrc, fcrc in enumerate(fcrcs):
        station_obj = m.get_station(fcrc.station_id, fcrc.survey_id)
        station_fc_group = station_obj.fourier_coefficients_group
        print(f"Available FC Groups for station {fcrc.station_id}: {station_fc_group.groups_list}")
        run_fc_group = station_obj.fourier_coefficients_group.get_fc_group(fcrc.run_id)
        print(run_fc_group)
        fc_dec_level = run_fc_group.get_decimation_level(fcrc.decimation_level_id)
        fc_dec_level_xrds = fc_dec_level.to_xarray()
        # could create name mapper dict from run_fc_group.channel_summary here if we wanted to.

        fc_dec_level = run_fc_group.get_decimation_level(decimation_level_id)
        fc_dec_level_xrds = fc_dec_level.to_xarray()

        name_dict = {f"{x}": f"{station_id}_{x}" for x in fc_dec_level_xrds.data_vars}
        print(name_dict)
        if i_station == 0:
            output = fc_dec_level_xrds.rename_vars(name_dict=name_dict)
        else:
            fc_dec_level_xrds = fc_dec_level_xrds.rename_vars(name_dict=name_dict)
            output = output.merge(fc_dec_level_xrds)

    return output
