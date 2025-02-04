"""
Definitions used in the creation of synthetic mth5 files.


Survey level: 'mth5_path', Path to output h5
Station level: 'station_id', name of the station
Station level:'latitude':17.996

Run level: 'columns', :channel names as a list; ["hx", "hy", "hz", "ex", "ey"]
Run level: 'raw_data_path', Path to ascii data source
Run level: 'noise_scalars', dict keyed by channel, default is zero,
Run level: 'nan_indices', iterable of integers, where to put nan [
Run level: 'filters', dict of filters keyed by columns
Run level: 'run_id', name of the run
Run level: 'sample_rate', 1.0

"""
import pathlib
from typing import Dict, List, Optional, Union

import mt_metadata.timeseries
import pandas as pd
from loguru import logger
from mt_metadata.timeseries.filters.helper_functions import make_coefficient_filter
from mt_metadata.timeseries import Run
from mt_metadata.timeseries import Station
from mt_metadata.transfer_functions.processing.aurora import ChannelNomenclature
from mt_metadata.transfer_functions.processing.aurora.channel_nomenclature import SupportedNomenclature

ASCII_DATA_PATH = pathlib.Path(__file__).parent.resolve()

def make_filters(as_list: Optional[bool] = False) -> Union[dict, list]:
    """
        Creates a collection of filters
    Because the synthetic data from EMTF are already in mV/km and nT, no calibration filters are required.
     The filters here are placeholders to show where instrument response function information would get assigned.

    :param as_list: If True we return a list, False return a dict
    :type as_list: bool
    :return filters_list: Filters for populating the filters lists of synthetic data
    :rtype filters_list: Union[List, Dict]
    """
    unity_coeff_filter = make_coefficient_filter(name="1", gain=1.0)
    multipy_by_10_filter = make_coefficient_filter(gain=10.0, name="10")
    divide_by_10_filter = make_coefficient_filter(gain=0.1, name="0.1")

    if as_list:
        return [unity_coeff_filter, multipy_by_10_filter, divide_by_10_filter]
    else:
        filters = {}
        filters["1x"] = unity_coeff_filter
        filters["10x"] = multipy_by_10_filter
        filters["0.1x"] = divide_by_10_filter
        return filters


FILTERS = make_filters()


class SyntheticRun(object):
    """
    Place to store information that will be needed to initialize and MTH5 Run object.

    Initially this class worked only with the synthetic ASCII data from legacy EMTF.
    """

    def __init__(
        self,
        id: str,
        sample_rate: Optional[float] = 1.0,
        raw_data_path: Optional[Union[str, pathlib.Path]] = None,
        channel_nomenclature: SupportedNomenclature = "default",
        channels: Optional[list] = None,
        noise_scalars: Optional[dict] = None,
        nan_indices: Optional[dict] = None,
        filters: Optional[dict] = None,
        start: Optional[str] = None,
        timeseries_dataframe: Optional[pd.DataFrame] = None,
        data_source: str = "legacy emtf ascii"
    ) -> None:
        """
        Constructor.

        :param id: label for the run
        :type id: str
        :param sample_rate: sample rate of the times series
        :type sample_rate: float
        :param raw_data_path: Path to ascii data source
        :type raw_data_path: Union[str, pathlib.Path, None]
        :param channel_nomenclature: the keyword for the channel nomenclature
        :type channel_nomenclature: str
        :param channels: the channel names to include in the run.
        :type channels: Union[list, None]
        :param noise_scalars: Keys are channels, values are scale factors for noise to add
        :type noise_scalars: Union[dict, None]
        :param nan_indices: Keys are channels, values lists.  List elements are pairs of (index, num_nan_to_add)
        :type nan_indices: Union[dict, None]
        :param filters: Keys are channels, values lists. List elements are Filter objects
        :type filters: Union[dict, None]
        :param start: Setting the run start time. e.g. start="1980-01-01T00:00:00+00:00"
        :type start: Union[str, None]
        :param timeseries_dataframe: The time series data for the run.
         Added 2025 to try to allow more general data to be cast to mth5
        :type timeseries_dataframe: Optional[pd.DataFrame] = None
        :param data_source: Keyword to tell if data are a legacy EMTF ASCII file
        :param data_source: Keyword to tell if data are a legacy EMTF ASCII file

        """
        run_metadata = Run()
        run_metadata.id = id
        run_metadata.sample_rate = sample_rate
        run_metadata.time_period.start = start

        self._timeseries_dataframe = timeseries_dataframe  # normally None for legacy EMTF data
        if isinstance(self._timeseries_dataframe, pd.DataFrame):
            self.data_source = "dataframe"
        else:
            self.data_source = data_source
            self.raw_data_path = raw_data_path

        # set channel names
        self._channel_map = None
        self.channel_nomenclature_keyword = channel_nomenclature
        self.set_channel_map()
        if channels is None:
            self.channels = list(self.channel_map.values())

        # Set scale factors for adding noise to individual channels
        self.noise_scalars = noise_scalars
        if noise_scalars is None:
            self.noise_scalars = {}
            for channel in self.channels:
                self.noise_scalars[channel] = 0.0

        # Set indices for adding nan to individual channels
        if nan_indices is None:
            self.nan_indices = {}  # TODO: make this consistent with noise_scalars, None or empty dict.

        # Set filters individual channels
        if filters is None:
            self.filters = {}  # TODO: make this consistent with noise_scalars, None or empty dict.

        # run_metadata.add_base_attribute("")
        self.run_metadata = run_metadata

    @property
    def channel_map(self) -> dict:
        """
        Make self._channel_map if it isn't initialize already.

        :rtype: dict
        :return: The mappings between the standard channel names and the ones that will be used in the MTH5.

        """
        if self._channel_map is None:
            self.set_channel_map()
        return self._channel_map

    def set_channel_map(self) -> None:
        """
        Populates a dictionary relating "actual" channel names to the standard names "hx", "hy", "hz", "ex", "ey"

        """
        channel_nomenclature = ChannelNomenclature(
            keyword=self.channel_nomenclature_keyword
        )
        self._channel_map = channel_nomenclature.get_channel_map()

    def _get_timeseries_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Returns time series data in a dataframe with columns named for EM field component.

        Up-samples data to run.sample_rate, which is treated as in integer.
        Only tested for 8, to make 8Hz data for testing.  If run.sample_rate is default (1.0)
        then no up-sampling takes place.

        :type run: mth5.data.station_config.SyntheticRun
        :param run: Information needed to define/create the run
        :type source_folder: Optional[Union[pathlib.Path, str]]
        :param source_folder: Where to load the ascii time series from.  This overwrites any
        previous value that may have been stored in the SyntheticRun
        :type add_nan_values: bool
        :param add_nan_values: If True, add some NaN, if False, do not add Nan.
        :rtype df: pandas.DataFrame
        :return df: The time series data for the synthetic run

        """
        if isinstance(self._timeseries_dataframe, pd.DataFrame):
            msg = f"Run Data appear to be already set in dataframe"
            logger.info(msg)
            return self._timeseries_dataframe

        elif self.data_source == "legacy emtf ascii":
            ascii_file = LegacyEMTFAsciiFile(file_path=self.raw_data_path)
            df = ascii_file.load_dataframe(channel_names=self.channels)
            return df
        else:
            msg = f"No dataframe associated with run, nor a legacy EMTF ASCII file"
            msg += ".. add support for your filetype or declare dataframe"
            raise NotImplementedError(msg)


class SyntheticStation(object):
    """
    Class used to contain information needed to generate MTH5 file from synthetic data.

    TODO: could add channel_nomenclature to this obj (instead of run) but would need to decide that
     runs cannot change channel nomenclature first. If that were decided, the channel_map() could go here as well.

    """

    def __init__(
        self,
        id: str,
        latitude: float = 0.0,
        mth5_name: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor.

        :param id: The name of the station
        :type id: str
        :param latitude: The station latitude
         TODO: Add note about units supported for latitude
         TODO: replace this with a station_metadata dictionary.
        :type latitude: float
        :type mth5_name: Union[str, pathlib.Path, None]
        :param mth5_name: The name of the h5 file to which the station data and metadata will be written.

        """
        self.id = id
        self.runs = []
        self.latitude = latitude
        self.mth5_name = mth5_name


def make_station_01(channel_nomenclature: SupportedNomenclature = "default") -> SyntheticStation:
    """
    :param channel_nomenclature: Must be one of the nomenclatures defined in SupportedNomenclature
    :type channel_nomenclature: str

    :return: Object with all info needed to generate MTH5 file from synthetic data.
    :rtype: SyntheticStation

    """
    station_metadata = Station()
    station_metadata.id = "test1"
    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature

    # initialize SyntheticStation
    station = SyntheticStation(station_metadata.id)
    station.mth5_name = f"{station_metadata.id}.h5"

    run_001 = SyntheticRun(
        id="001",
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        channel_nomenclature=channel_nomenclature,
        start=None,
    )

    # assign indices to set to Nan (not used 2024-06-06)
    nan_indices = {}
    for ch in run_001.channels:
        nan_indices[ch] = []
        if ch == channel_nomenclature_obj.hx:
            nan_indices[ch].append([11, 100])
        if ch == channel_nomenclature_obj.hy:
            nan_indices[ch].append([11, 100])
            nan_indices[ch].append([20000, 444])
    run_001.nan_indices = nan_indices

    # assign some filters to the channels
    filters = {}
    for ch in run_001.channels:
        if ch in channel_nomenclature_obj.ex_ey:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in channel_nomenclature_obj.hx_hy_hz:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]
    run_001.filters = filters

    station.runs = [
        run_001,
    ]
    station_metadata.run_list = [
        run_001,
    ]
    station.station_metadata = station_metadata
    return station


def make_station_02(channel_nomenclature: SupportedNomenclature = "default") -> SyntheticStation:
    """
    Just like station 1, but the data are different

    :param channel_nomenclature: Must be one of the nomenclatures defined in SupportedNomenclature
    :type channel_nomenclature: SupportedNomenclature
    :return: Object with all info needed to generate MTH5 file from synthetic data.
    :rtype: SyntheticStation

    """
    test2 = make_station_01(channel_nomenclature=channel_nomenclature)
    test2.id = "test2"
    test2.mth5_name = "test2.h5"
    test2.runs[0].raw_data_path = ASCII_DATA_PATH.joinpath("test2.asc")
    nan_indices = {}
    for channel in test2.runs[0].channels:
        nan_indices[channel] = []
    test2.runs[0].nan_indices = nan_indices
    return test2


def make_station_03(channel_nomenclature: SupportedNomenclature = "default") -> SyntheticStation:
    """
    Create a synthetic station with multiple runs.  Rather than generate fresh
    synthetic data, we just reuse test1.asc for each run.

    :param channel_nomenclature: Literal, Must be one of the nomenclatures defined in "channel_nomenclatures.json"
    :type channel_nomenclature: SupportedNomenclature
    :rtype: SyntheticStation
    :return: Object with all info needed to generate MTH5 file from synthetic data.

    """
    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature
    station = SyntheticStation("test3")
    station.mth5_name = "test3.h5"
    channels = channel_nomenclature_obj.channels

    nan_indices = {}
    for ch in channels:
        nan_indices[ch] = []

    filters = {}
    for ch in channels:
        if ch in channel_nomenclature_obj.ex_ey:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in channel_nomenclature_obj.hx_hy_hz:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]

    run_001 = SyntheticRun(
        "001",
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        nan_indices=nan_indices,
        filters=filters,
        channel_nomenclature=channel_nomenclature,
        start="1980-01-01T00:00:00+00:00",
    )

    noise_scalars = {}
    for ch in channels:
        noise_scalars[ch] = 2.0
    run_002 = SyntheticRun(
        "002",
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
        channel_nomenclature=channel_nomenclature,
        start="1980-01-02T00:00:00+00:00",
    )

    for ch in channels:
        noise_scalars[ch] = 5.0
    run_003 = SyntheticRun(
        "003",
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
        channel_nomenclature=channel_nomenclature,
        start="1980-01-03T00:00:00+00:00",
    )

    for ch in channels:
        noise_scalars[ch] = 10.0
    run_004 = SyntheticRun(
        "004",
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
        channel_nomenclature=channel_nomenclature,
        start="1980-01-04T00:00:00+00:00",
    )

    run_001.filters = filters
    run_002.filters = filters
    run_003.filters = filters
    run_004.filters = filters

    station.runs = [run_001, run_002, run_003, run_004]

    return station


def make_station_04(channel_nomenclature: SupportedNomenclature = "default") -> SyntheticStation:
    """
    Just like station 01, but data are resampled to 8Hz

    :param channel_nomenclature: Literal, Must be one of the nomenclatures defined in "channel_nomenclatures.json"
    :type channel_nomenclature: SupportedNomenclature
    :rtype: SyntheticStation
    :return: Object with all info needed to generate MTH5 file from synthetic data.
    """
    station_metadata = Station()
    station_metadata.id = "test1"
    channel_nomenclature_obj = ChannelNomenclature()
    channel_nomenclature_obj.keyword = channel_nomenclature

    station = SyntheticStation("test1")
    station.mth5_name = "test_04_8Hz.h5"

    run_001 = SyntheticRun(
        "001",
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        channel_nomenclature=channel_nomenclature,
        start=None,
        sample_rate=8.0,
    )
    run_001.nan_indices = {}

    filters = {}
    for ch in run_001.channels:
        if ch in channel_nomenclature_obj.ex_ey:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in channel_nomenclature_obj.hx_hy_hz:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]
    run_001.filters = filters

    station.runs = [
        run_001,
    ]
    station_metadata.run_list = [
        run_001,
    ]
    station.station_metadata = station_metadata
    return station


class LegacyEMTFAsciiFile():
    """
        This class can be used to interact with the legacy synthetic data files
        that were originally in EMTF.

    """
    def __init__(
        self,
        file_path: pathlib.Path
    ):
        self.file_path  = file_path

    def load_dataframe(self, channel_names: list) -> pd.DataFrame:
        """
            Loads an EMTF legacy ASCII time series into a dataframe.

            These files have an awkward whitespace separator, and also need to have the
             electric field channels inverted to fix a phase swap.

            :param channel_names: The names of the channels in the legacy EMTF file, in order.
            :type channel_names: list
            :return df: The labelled time series from the legacy EMTF file.
            :rtype df: pd.DataFrame

        """

        # read in data
        df = pd.read_csv(self.file_path, names=channel_names, sep="\s+")

        # Invert electric channels to fix phase swap due to modeling coordinates.
        # Column indices are used to avoid handling channel nomenclature here.
        df[df.columns[-2]] = -df[df.columns[-2]]  # df["ex"] = -df["ex"]
        df[df.columns[-1]] = -df[df.columns[-1]]  # df["ey"] = -df["ey"]

        return df

def main():
    # sr = SyntheticRun("001")
    make_station_04()


if __name__ == "__main__":
    main()
