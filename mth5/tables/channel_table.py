# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:09:38 2022

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import numpy as np
import h5py

from mth5 import CHANNEL_DTYPE
from mth5.tables import MTH5Table

# =============================================================================


class ChannelSummaryTable(MTH5Table):
    """
    Object to hold the channel summary and provide some convenience functions
    like fill, to_dataframe ...
    
    """

    def __init__(self, hdf5_dataset):
        super().__init__(hdf5_dataset)
        self._dtype = CHANNEL_DTYPE

    def to_dataframe(self):
        """
        Create a pandas DataFrame from the table for easier querying.
        
        :return: Channel Summary 
        :rtype: :class:`pandas.DataFrame`

        """

        df = pd.DataFrame(self.array[()])
        for key in [
            "survey",
            "station",
            "run",
            "component",
            "measurement_type",
            "units",
        ]:
            setattr(df, key, getattr(df, key).str.decode("utf-8"))
        df.start = pd.to_datetime(df.start.str.decode("utf-8"))
        df.end = pd.to_datetime(df.end.str.decode("utf-8"))

        return df

    def summarize(self):
        """
        
        :return: DESCRIPTION
        :rtype: TYPE

        """

        def recursive_get_channel_entry(group):
            """
            a function to get channel entry
            """
            if isinstance(group, (h5py._hl.group.Group, h5py._hl.files.File)):
                for key, node in group.items():
                    recursive_get_channel_entry(node)
            elif isinstance(group, h5py._hl.dataset.Dataset):
                try:
                    ch_type = group.attrs["type"]
                    if ch_type in ["electric", "magnetic", "auxiliary"]:
                        ch_entry = np.array(
                            [
                                (
                                    group.parent.parent.parent.parent.attrs["id"],
                                    group.parent.parent.attrs["id"],
                                    group.parent.attrs["id"],
                                    group.parent.parent.attrs["location.latitude"],
                                    group.parent.parent.attrs["location.longitude"],
                                    group.parent.parent.attrs["location.elevation"],
                                    group.attrs["component"],
                                    group.attrs["time_period.start"],
                                    group.attrs["time_period.end"],
                                    group.size,
                                    group.attrs["sample_rate"],
                                    group.attrs["type"],
                                    group.attrs["measurement_azimuth"],
                                    group.attrs["measurement_tilt"],
                                    group.attrs["units"],
                                    group.ref,
                                    group.parent.ref,
                                    group.parent.parent.ref,
                                )
                            ],
                            dtype=CHANNEL_DTYPE,
                        )
                        self.add_row(ch_entry)

                except KeyError as error:
                    # self.logger.exception(error)
                    # return None
                    pass

        recursive_get_channel_entry(self.array.parent)
