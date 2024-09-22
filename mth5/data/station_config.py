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
from mt_metadata.timeseries.filters.helper_functions import make_coefficient_filter
from mt_metadata.timeseries import Run
from mt_metadata.timeseries import Station
from mt_metadata.transfer_functions.processing.aurora import ChannelNomenclature

ASCII_DATA_PATH = pathlib.Path(__file__).parent.resolve()

def make_filters(as_list: Optional[bool] = False) -> Union[dict, list]:
    """
    Because the data from EMTF is already in mV/km and nT these filters are just
    placeholders to show where they would get assigned.

    :type as_list: bool
    :param as_list: If True we return a list, False return a dict
    :rtype filters_list: Union[List, Dict]
    :return pfilters_list: Filters for populating the filters lists of synthetic data
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
        raw_data_path: Optional[Union[str, pathlib.Path, None]] = None,
        channel_nomenclature: Optional[str] = "default",
        channels: Optional[Union[list, None]] = None,
        noise_scalars: Optional[Union[dict, None]] = None,
        nan_indices: Optional[Union[dict, None]] = None,
        filters: Optional[Union[dict, None]] = None,
        start: Optional[Union[str, None]] = None,
    ) -> None:
        """
        Constructor.

        :type id: str
        :param id: label for the run
        :type sample_rate: float
        :param sample_rate: sample rate of the times series
        :type raw_data_path: Union[str, pathlib.Path, None]
        :param raw_data_path: Path to ascii data source
        :type channel_nomenclature: str
        :param channel_nomenclature: the keyword for the channel nomenclature
        :type channels: Union[list, None]
        :param channels: the channel names to include in the run.
        :type noise_scalars: Union[dict, None]
        :param noise_scalars: Keys are channels, values are scale factors for noise to add
        :type nan_indices: Union[dict, None]
        :param nan_indices: Keys are channels, values lists.  List elements are pairs of (index, num_nan_to_add)
        :type filters: Union[dict, None]
        :param filters: Keys are channels, values lists. List elements are Filter objects
        :type start: Union[str, None]
        :param start: Setting the run start time. e.g. start="1980-01-01T00:00:00+00:00"

        """
        run_metadata = Run()
        run_metadata.id = id
        run_metadata.sample_rte = sample_rate

        self.raw_data_path = raw_data_path

        # set channel names
        self._channel_map = None
        self.channel_nomenclature_keyword = channel_nomenclature
        self.set_channel_map()
        if channels is None:
            self.channels = list(self.channel_map.values())
        self.noise_scalars = noise_scalars
        if noise_scalars is None:
            self.noise_scalars = {}
            for channel in self.channels:
                self.noise_scalars[channel] = 0.0
        if nan_indices is None:
            self.nan_indices = {}  # TODO: make this consistent with noise_scalars, None or empty dict.
        if filters is None:
            self.filters = {}  # TODO: make this consistent with noise_scalars, None or empty dict.
        self.start = start

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


class SyntheticStation(object):
    """
    Class used to contain information needed to generate MTH5 file from synthetic data.

    TODO: could add channel_nomenclature to this obj (instead of run) but would need to decide that
     runs cannot change channel nomenclature first. If that were decided, the channel_map() could go here as well.

    """

    def __init__(
        self,
        id: str,
        latitude: Optional[float] = 0.0,
        mth5_name: Optional[Union[str, pathlib.Path, None]] = None,
    ) -> None:
        """
        Constructor.

        :type id: str
        :param id:: station id
        :type latitude: float
        :param latitude: the station latiude
        :type mth5_name: Union[str, pathlib.Path, None]
        :param mth5_name: The name of thm mth5 the station will be written to.

        """
        self.id = id
        self.runs = []
        self.latitude = latitude
        self.mth5_name = mth5_name


def make_station_01(channel_nomenclature: Optional[str] = "default") -> SyntheticStation:
    """

    :type channel_nomenclature: str
    :param channel_nomenclature: Must be one of the nomenclatures defined in "channel_nomenclatures.json"

    :rtype: SyntheticStation
    :return: Object with all info needed to generate MTH5 file from synthetic data.

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


def make_station_02(channel_nomenclature: Optional[str] = "default") -> SyntheticStation:
    """
    Just like station 1, but the data are different

    :type channel_nomenclature: str
    :param channel_nomenclature: Must be one of the nomenclatures defined in "channel_nomenclatures.json"
    :rtype: SyntheticStation
    :return: Object with all info needed to generate MTH5 file from synthetic data.

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


def make_station_03(channel_nomenclature="default") -> SyntheticStation:
    """
    Create a synthetic station with multiple runs.  Rather than generate fresh
    synthetic data, we just reuse test1.asc for each run.

    :type channel_nomenclature: str
    :param channel_nomenclature: Must be one of the nomenclatures defined in "channel_nomenclatures.json"
    Example values ["default", "lemi12", "lemi34", "phoenix123"]
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


def make_station_04(channel_nomenclature="default") -> SyntheticStation:
    """
    Just like station 01, but data are resampled to 8Hz

    :type channel_nomenclature: str
    :param channel_nomenclature: Must be one of the nomenclatures defined in "channel_nomenclatures.json"
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


def main():
    # sr = SyntheticRun("001")
    make_station_04()


if __name__ == "__main__":
    main()
