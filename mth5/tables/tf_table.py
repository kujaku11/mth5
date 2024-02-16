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

from mth5 import TF_DTYPE
from mth5.tables import MTH5Table
from mth5.helpers import validate_name

# =============================================================================


class TFSummaryTable(MTH5Table):
    """
    Object to hold the channel summary and provide some convenience functions
    like fill, to_dataframe ...

    """

    def __init__(self, hdf5_dataset):
        super().__init__(hdf5_dataset, TF_DTYPE)

    def to_dataframe(self):
        """
        Create a pandas DataFrame from the table for easier querying.

        :return: Channel Summary
        :rtype: :class:`pandas.DataFrame`

        """

        df = pd.DataFrame(self.array[()])
        for key in [
            "station",
            "survey",
            "tf_id",
            "units",
        ]:
            setattr(df, key, getattr(df, key).str.decode("utf-8"))
        return df

    def summarize(self):
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.clear_table()

        def recursive_get_tf_entry(group):
            """
            a function to get tf entry, hopefully this is faster than looping
            and getting the correct group object.

            """
            if isinstance(group, (h5py._hl.group.Group, h5py._hl.files.File)):
                for key, node in group.items():
                    try:
                        group_type = node.attrs["mth5_type"].lower()
                        if group_type == "transferfunction":
                            has_impedance = False
                            has_tipper = False
                            has_covariance = False
                            if "transfer_function" in node.keys():
                                tf_dataset = node["transfer_function"]
                                if tf_dataset != (1, 1, 1):
                                    nz = np.nonzero(tf_dataset)
                                    unique_values = np.unique(nz[1])
                                    if (
                                        0 in unique_values
                                        or 1 in unique_values
                                    ):
                                        has_impedance = True
                                    if 2 in unique_values:
                                        has_tipper = True
                            if (
                                "residual_covariance" in node.keys()
                                and "inverse_signal_power" in node.keys()
                            ):
                                res = node["residual_covariance"]
                                isp = node["inverse_signal_power"]

                                if res.shape != (1, 1, 1) and isp.shape != (
                                    1,
                                    1,
                                    1,
                                ):
                                    has_covariance = True
                            if "period" in node.keys():
                                period = node["period"][()]
                            else:
                                period = np.zeros(2)
                            tf_entry = np.array(
                                [
                                    (
                                        validate_name(
                                            node.parent.parent.attrs["id"]
                                        ),
                                        validate_name(
                                            node.parent.parent.parent.parent.attrs[
                                                "id"
                                            ]
                                        ),
                                        node.parent.parent.attrs[
                                            "location.latitude"
                                        ],
                                        node.parent.parent.attrs[
                                            "location.longitude"
                                        ],
                                        node.parent.parent.attrs[
                                            "location.elevation"
                                        ],
                                        validate_name(node.attrs["id"]),
                                        node.attrs["units"],
                                        has_impedance,
                                        has_tipper,
                                        has_covariance,
                                        period.min(),
                                        period.max(),
                                        node.ref,
                                        node.parent.parent.ref,
                                    )
                                ],
                                dtype=TF_DTYPE,
                            )
                            self.add_row(tf_entry)
                        else:
                            recursive_get_tf_entry(node)
                    except KeyError:
                        recursive_get_tf_entry(node)
            elif isinstance(group, h5py._hl.dataset.Dataset):
                pass

        recursive_get_tf_entry(self.array.parent)
