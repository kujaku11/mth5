# -*- coding: utf-8 -*-
from __future__ import annotations


"""Reports group utilities for storing report and image artifacts in MTH5."""

from pathlib import Path
from typing import Any

import h5py

# =============================================================================
# Imports
# =============================================================================
import numpy as np
from PIL import Image

from mth5.groups.base import BaseGroup


# =============================================================================
# Reports Group
# =============================================================================
class ReportsGroup(BaseGroup):
    """Store report files (PDF/text) and images under ``/Survey/Reports``.

    Files are embedded into HDF5 datasets with basic metadata preserved.

    Examples
    --------
    >>> reports = survey.reports_group
    >>> _ = reports.add_report("site_report", filename="/tmp/report.pdf")
    >>> _ = reports.get_report("site_report")  # doctest: +SKIP
    """

    def __init__(self, group: h5py.Group, **kwargs: Any) -> None:
        super().__init__(group, **kwargs)
        self._accepted_reports: list[str] = ["pdf", "txt", "md"]
        self._accepted_images: list[str] = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]

        # summary of reports
        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (1000,),
            "dtype": np.dtype(
                [
                    ("name", "S5"),
                    ("type", "S32"),
                    ("summary", "S200"),
                    ("creation_time", "S32"),
                    ("hdf5_reference", h5py.ref_dtype),
                ]
            ),
        }

    def add_report(
        self,
        report_name: str,
        report_metadata: dict[str, Any] | None = None,
        filename: str | Path | None = None,
    ) -> None:
        """Add a report or image file to the group.

        Parameters
        ----------
        report_name : str
            Dataset name to store the file under.
        report_metadata : dict, optional
            Additional attributes to attach to the dataset.
        filename : str or Path, optional
            Path to the file to embed; supported types: PDF/TXT/MD and common images.

        Raises
        ------
        FileNotFoundError
            If ``filename`` does not exist.

        Examples
        --------
        >>> reports.add_report("manual", filename="docs/manual.pdf")  # doctest: +SKIP
        """

        if filename is not None:
            filename = Path(filename)
            if not filename.exists():
                raise FileNotFoundError(f"{filename} does not exist")
            extension = filename.suffix.lower()[1:]
            if extension in self._accepted_reports:
                data_bytes = filename.read_bytes()

            elif extension in self._accepted_images:
                # Open image and convert to numpy array
                img = Image.open(filename)
                data_bytes = np.array(img)

            else:
                self.logger.error(
                    f"Adding files of type {extension} is not implemented yet"
                )

            # Save image data into HDF5
            dataset = self.hdf5_group.create_dataset(report_name, data=data_bytes)

            # Add metadata if provided
            if report_metadata is not None:
                for key, value in report_metadata.items():
                    dataset.attrs[key] = value
            else:
                dataset.attrs["description"] = f"{extension.upper()} image file"
                dataset.attrs["filename"] = filename.name
                dataset.attrs["file_type"] = extension
                dataset.attrs["creation_time"] = str(filename.stat().st_ctime)

    def get_report(self, report_name: str) -> Path:
        """Extract a stored report or image to the current working directory.

        Parameters
        ----------
        report_name : str
            Name of the stored dataset.

        Returns
        -------
        pathlib.Path
            Path to the materialized file on disk.

        Raises
        ------
        ValueError
            If the stored file type is unsupported.

        Examples
        --------
        >>> path = reports.get_report("site_report")  # doctest: +SKIP
        >>> path.exists()
        True
        """

        dataset = self.hdf5_group[report_name]
        file_type = dataset.attrs["file_type"]

        print(f"DEBUG: dataset content before bytes conversion: {dataset[()]}")

        if file_type in self._accepted_reports:
            report_data = bytes(dataset[()])
            fn_path = Path().cwd().joinpath(dataset.attrs["filename"])
            fn_path.write_bytes(report_data)
            self.logger.info(f"Report extracted to {fn_path}")
            return fn_path

        if file_type in self._accepted_images:
            img_data = np.array(dataset[()])
            img = Image.fromarray(img_data)
            fn_path = Path().cwd().joinpath(dataset.attrs["filename"])
            img.save(fn_path)
            self.logger.info(f"Image report extracted to {fn_path}")
            return fn_path

        raise ValueError(f"Unsupported file type '{file_type}' for {report_name}")

    def list_reports(self) -> list[str]:
        """List all stored reports and images in the group.

        Returns
        -------
        list of str
            Names of all stored datasets in the reports group.

        Examples
        --------
        >>> report_names = reports.list_reports()  # doctest: +SKIP
        >>> print(report_names)
        ['site_report', 'manual', 'overview_image']
        """
        return list(self.hdf5_group.keys())

    def remove_report(self, report_name: str) -> None:
        """Remove a stored report or image from the group.

        Parameters
        ----------
        report_name : str
            Name of the stored dataset to remove.

        Examples
        --------
        >>> reports.remove_report("manual")  # doctest: +SKIP
        """
        if report_name in self.hdf5_group:
            del self.hdf5_group[report_name]
            self.logger.info(f"Removed report '{report_name}' from the group.")
        else:
            self.logger.warning(f"Report '{report_name}' not found in the group.")
