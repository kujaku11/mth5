"""
Convert MTH5 to other formats

- MTH5 -> miniSEED + StationXML
"""

# ==================================================================================
# Imports
# ==================================================================================
import re
from pathlib import Path
from mth5.mth5 import MTH5
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment

# ==================================================================================


class MTH5ToMiniSEEDStationXML:
    def __init__(
        self, mth5_path, save_path=None, network_code=None, use_runs_with_data_only=True
    ):
        self.mth5_path = mth5_path
        self.save_path = save_path
        self.network_code = network_code
        self.use_runs_with_data_only = use_runs_with_data_only

        self._network_code_pattern = r"^[a-zA-Z0-9]{2}$"

    @property
    def mth5_path(self):
        """Path to MTH5 file"""
        return self._mth5_path

    @mth5_path.setter
    def mth5_path(self, value):
        """make sure mth5 path exists"""
        try:
            value = Path(value)
        except Exception as error:
            raise TypeError(f"Could not convert value to Path: {error}")

        if not value.exists():
            raise FileExistsError(f"Could not find {value}")

        self._mth5_path = value

    @property
    def save_path(self):
        """path to save miniSEED and StationXML to"""
        return self._save_path

    @save_path.setter
    def save_path(self, value):
        """Set the save path, if None set to parent directory of mth5_path"""
        if value is None:
            self._save_path = self._mth5_path.parent
        else:
            value = Path(value)

        if not self._save_path.exists():
            self._save_path.mkdir(exists_ok=True)

    @property
    def network_code(self):
        """2 alpha-numeric string provided by FDSN DMC"""
        return self._network_code

    @network_code.setter
    def network_code(self, value):
        """be sure the string is just 2 characters"""
        if not re.match(pattern, value):
            raise ValueError(
                f"{value} is not a valid network code. It must be 2 alphanumeric characters"
            )
        self._nework_code = value
