# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:57:54 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from mth5.mth5 import MTH5
from mth5 import read_file
from mth5.io.lemi import LEMI424, LEMICollection

# =============================================================================


class LEMI424Client:
    def __init__(
        self,
        data_path,
        file_ext=["txt", "TXT"],
        save_path=None,
        mth5_filename=None,
    ):
        self.data_path = data_path
        self.file_extensions = file_ext
        self.save_path = save_path
        self.mth5_filename = mth5_filename
        self.sample_rate = [1]

    @property
    def data_path(self):
        """Path to lemi424 data"""
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        """

        :param value: data path, directory to where files are
        :type value: str or Path

        """

        if value is not None:
            self._data_path = Path(value)
            if not self._data_path.exists():
                raise IOError(f"Could not find {self._data_path}")

            self.collection = LEMICollection(self.data_path)

        else:
            raise ValueError("data_path cannot be None")

    @property
    def save_path(self):
        """Path to save mth5"""
        return self._save_path.joinpath(self.mth5_filename)

    @save_path.setter
    def save_path(self, value):
        """

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if value is not None:
            value = Path(value)
            if value.is_dir():
                self._save_path = value
            else:
                self._save_path = value.parent
                self.mth5_filename = value.name

        else:
            self._save_path = self.data_path

    def get_run_dict(self):
        """

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.collection.get_runs(sample_rates=self.sample_rate)
