# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:03:53 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

from pathlib import Path

import h5py

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from mth5.groups.base import BaseGroup


# =============================================================================
# Reports Group
# =============================================================================
class ReportsGroup(BaseGroup):
    """
    Not sure how to handle this yet

    """

    def __init__(self, group, **kwargs):
        super().__init__(group, **kwargs)

        # summary of reports
        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (1000,),
            "dtype": np.dtype(
                [
                    ("name", "S5"),
                    ("type", "S32"),
                    ("summary", "S200"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        }

    def add_report(self, report_name, report_metadata=None, report_filename=None):
        """

        :param report_name: DESCRIPTION
        :type report_name: TYPE
        :param report_metadata: DESCRIPTION, defaults to None
        :type report_metadata: TYPE, optional
        :param report_data: DESCRIPTION, defaults to None
        :type report_data: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if report_filename is not None:
            report_filename = Path(report_filename)
            if not report_filename.exists():
                raise FileNotFoundError(f"{report_filename} does not exist")
            extension = report_filename.suffix.lower()
            if extension == ".pdf":
                # Read PDF as binary
                with open(report_filename, "rb") as f:
                    pdf_bytes = f.read()

                # Save PDF bytes into HDF5
                dataset = self.hdf5_group.create_dataset(
                    "pdf_file", data=np.void(pdf_bytes)
                )
                # Add metadata if provided
                if report_metadata is not None:
                    for key, value in report_metadata.items():
                        dataset.attrs[key] = value
                else:
                    dataset.attrs["description"] = "PDF report file"
                    dataset.attrs["filename"] = report_filename.name
                    dataset.attrs["file_type"] = "pdf"
            else:
                self.logger.error(
                    f"Adding files of type {extension} is not implemented yet"
                )

    def get_report(self, report_name):
        """

        :param report_name: DESCRIPTION
        :type report_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        dataset = self.hdf5_group[report_name]
        if dataset.attrs["file_type"] == "pdf":
            pdf_data = bytes(dataset[()])
            fn_path = Path().cwd().joinpath(dataset.attrs["filename"])
            with open(fn_path, "wb") as f:
                f.write(pdf_data)
            self.logger.info(f"PDF report written to {fn_path}")
            return fn_path
