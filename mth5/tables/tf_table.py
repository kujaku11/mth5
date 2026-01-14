# -*- coding: utf-8 -*-
"""
Transfer function summary table utilities.

Summarize `TransferFunction` groups stored in an MTH5 file into a structured
table and provide a convenient `pandas.DataFrame` view for querying.

Notes
-----
- Traversal searches for groups with attribute ``mth5_type='transferfunction'``
    and collects basic availability flags (impedance, tipper, covariance) along
    with period range and references.

"""

from __future__ import annotations

import h5py
import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pandas as pd

from mth5 import TF_DTYPE
from mth5.helpers import validate_name
from mth5.tables import MTH5Table


# =============================================================================


class TFSummaryTable(MTH5Table):
    """
    Summary table for `TransferFunction` groups.

    Provides convenience functions to populate the table (`summarize`) and
    export to `pandas.DataFrame` (`to_dataframe`).

    Examples
    --------
    Build and export a TF summary::

        >>> import h5py
        >>> from mth5.tables.tf_table import TFSummaryTable
        >>> f = h5py.File('example.mth5', 'r')
        >>> tf_summary_ds = f['Exchange']['TF_Summary']
        >>> tf_table = TFSummaryTable(tf_summary_ds)
        >>> tf_table.summarize()
        >>> df = tf_table.to_dataframe()
        >>> df.head()
    """

    def __init__(self, hdf5_dataset: h5py.Dataset) -> None:
        super().__init__(hdf5_dataset, TF_DTYPE)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the table to a `pandas.DataFrame` for easier querying.

        Returns
        -------
        pandas.DataFrame
            A dataframe with decoded string columns.

        Examples
        --------
        Filter transfer functions that include tipper::

            >>> df = tf_table.to_dataframe()
            >>> df[df.has_tipper]
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

    def summarize(self) -> None:
        """
        Populate the summary table by traversing the HDF5 hierarchy.

        Searches for groups where ``mth5_type`` equals ``'transferfunction'``
        and adds a row indicating available datasets (impedance, tipper,
        covariance), period min/max, and relevant references.

        Returns
        -------
        None

        Examples
        --------
        Refresh the TF summary::

            >>> tf_table.clear_table()
            >>> tf_table.summarize()
        """
        self.clear_table()

        def recursive_get_tf_entry(
            group: h5py.Group | h5py.File | h5py.Dataset,
        ) -> None:
            """Recursively collect TF summary entries from the hierarchy."""
            if isinstance(group, (h5py.Group, h5py.File)):
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
                                    if 0 in unique_values or 1 in unique_values:
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
                                        ).encode("utf-8"),
                                        validate_name(
                                            node.parent.parent.parent.parent.attrs["id"]
                                        ).encode("utf-8"),
                                        node.parent.parent.attrs["location.latitude"],
                                        node.parent.parent.attrs["location.longitude"],
                                        node.parent.parent.attrs["location.elevation"],
                                        validate_name(node.attrs["id"]).encode("utf-8"),
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
            elif isinstance(group, h5py.Dataset):
                pass

        parent = self.array.parent
        if not isinstance(parent, (h5py.Group, h5py.File)):
            raise TypeError("Unexpected parent type for summary dataset.")
        recursive_get_tf_entry(parent)
