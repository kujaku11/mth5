# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:36:26 2024

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

from loguru import logger


# =============================================================================


class ClientBase:
    def __init__(
        self,
        data_path,
        sample_rates=[1],
        save_path=None,
        mth5_filename="from_client.h5",
        **kwargs,
    ):
        """

        any h5 parameters should be in kwargs as `h5_parameter_name`

        :param data_path: DESCRIPTION
        :type data_path: TYPE
        :param sample_rate: DESCRIPTION, defaults to [1]
        :type sample_rate: TYPE, optional
        :param save_path: DESCRIPTION, defaults to None
        :type save_path: TYPE, optional
        :param mth5_filename: DESCRIPTION, defaults to "from_client.h5"
        :type mth5_filename: TYPE, optional
        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :param : DESCRIPTION
        :type : TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.logger = logger

        self.data_path = data_path
        self.mth5_filename = mth5_filename
        self.sample_rates = sample_rates
        self.save_path = save_path

        self.interact = False

        self.mth5_version = "0.2.0"
        self.h5_compression = "gzip"
        self.h5_compression_opts = 4
        self.h5_shuffle = True
        self.h5_fletcher32 = True
        self.h5_data_level = 1
        self.mth5_file_mode = "w"

        self.collection = None

        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    @property
    def h5_kwargs(self):
        h5_params = dict(
            file_version=self.mth5_version,
            compression=self.h5_compression,
            compression_opts=self.h5_compression_opts,
            shuffle=self.h5_shuffle,
            fletcher32=self.h5_fletcher32,
            data_level=self.h5_data_level,
        )

        for key, value in self.__dict__.items():
            if key.startswith("h5"):
                h5_params[key[3:]] = value

        return h5_params

    @property
    def data_path(self):
        """Path to data"""
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

        else:
            raise ValueError("data_path cannot be None")

    @property
    def sample_rates(self):
        """sample rates to look for"""
        return self._sample_rates

    @sample_rates.setter
    def sample_rates(self, value):
        """
        sample rates set to a list

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if isinstance(value, (int, float)):
            self._sample_rates = [value]
        elif isinstance(value, str):
            self._sample_rates = [float(v) for v in value.split(",")]

        elif isinstance(value, (tuple, list)):
            self._sample_rates = [float(v) for v in value]
        else:
            raise TypeError(f"Cannot parse {type(value)}")

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
            if value.exists():
                if value.is_dir():
                    self._save_path = value
                elif value.is_file():
                    self._save_path = value.parent
                    self.mth5_filename = value.name
            else:
                if "." in value.name:
                    self._save_path = value.parent
                    self.mth5_filename = value.name
                else:
                    self._save_path = value
                self._save_path.mkdir(exist_ok=True)
        else:
            self._save_path = self.data_path

    def get_run_dict(self):
        """
        Get Run information

        :return: DESCRIPTION
        :rtype: TYPE

        """

        return self.collection.get_runs(sample_rates=self.sample_rates)
