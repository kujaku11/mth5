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
from loguru import logger

# ==================================================================================


class MTH5ToMiniSEEDStationXML:
    def __init__(
        self,
        mth5_path=None,
        save_path=None,
        network_code="ZU",
        use_runs_with_data_only=True,
        **kwargs,
    ):
        self._network_code_pattern = r"^[a-zA-Z0-9]{2}$"
        self.mth5_path = mth5_path
        self.save_path = save_path
        self.network_code = network_code
        self.use_runs_with_data_only = use_runs_with_data_only
        self.encoding = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def mth5_path(self):
        """Path to MTH5 file"""
        return self._mth5_path

    @mth5_path.setter
    def mth5_path(self, value):
        """make sure mth5 path exists"""
        if value is None:
            self._mth5_path = None
            return
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
            if self._mth5_path is None:
                self._save_path = Path().cwd()
            else:
                self._save_path = self._mth5_path.parent
        else:
            self._save_path = Path(value)

        if not self._save_path.exists():
            self._save_path.mkdir(exists_ok=True)

    @property
    def network_code(self):
        """alpha-numeric string of 2 characters provided by FDSN DMC"""
        return self._network_code

    @network_code.setter
    def network_code(self, value):
        """be sure the string is just 2 characters"""
        if value is None:
            raise ValueError(
                "Must input a network code.  "
                "Request a temporary code from https://www.fdsn.org/networks/request/temp/"
            )
        if not re.match(self._network_code_pattern, value):
            raise ValueError(
                f"{value} is not a valid network code. It must be 2 alphanumeric characters"
            )
        self._network_code = value

    @classmethod
    def convert_mth5_to_ms_stationxml(
        cls,
        mth5_path,
        save_path=None,
        network_code="ZU",
        use_runs_with_data_only=True,
        **kwargs,
    ):
        """Class method to convert an MTH5 to miniSEED and StationXML"""

        converter = cls(
            mth5_path=mth5_path,
            save_path=save_path,
            network_code=network_code,
            use_runs_with_data_only=use_runs_with_data_only,
            **kwargs,
        )

        with MTH5() as m:
            m.open_mth5(converter.mth5_path)
            experiment = m.to_experiment(has_data=converter.use_runs_with_data_only)
            stream_list = []
            for row in m.run_summary.itertuples():
                if row.has_data:
                    run_ts = m.from_reference(row.run_hdf5_reference).to_runts()
                    if converter.encoding is None:
                        encoding = get_encoding(run_ts)
                    else:
                        encoding = converter.encoding
                    stream = run_ts.to_obspy_stream(
                        network_code=converter.network_code, encoding=encoding
                    )
                    stream_fn = converter.save_path.joinpath(
                        f"{row.survey}_{row.station}_{row.run}.mseed"
                    )
                    stream.write(
                        stream_fn,
                        format="MSEED",
                        reclen=256,
                        encoding=encoding,
                    )
                    logger.info(
                        f"Wrote miniSEED for {row.survey}.{row.station}.{row.run} to {stream_fn}"
                    )
                    stream_list.append(stream_fn)

        # write StationXML
        experiment.surveys[0].fdsn.network = converter.network_code

        translator = XMLInventoryMTExperiment()
        xml_fn = converter.save_path.joinpath(f"{converter.mth5_path.stem}.xml")
        stationxml = translator.mt_to_xml(
            experiment,
            stationxml_fn=xml_fn,
        )
        logger.info(f"Wrote StationXML to {xml_fn}")

        return xml_fn, stream_list


def get_encoding(run_ts):
    """
    need to make sure that the encoding for each run is the same across channels, or there may be
    compatibility issues with the miniseed files.

    For now take the median dtype
    """
    dtypes = [run_ts.dataset[ch].data.dtype.name for ch in run_ts.channels]
    encoding = sorted(dtypes)[int(len(dtypes) / 2)].upper()
    if encoding in ["INT64"]:
        encoding = "INT32"
        logger.warning("Casting INT64 to INT32")

    return encoding
