"""

This module contains tools for building MTH5 files from synthetic data.

Development Notes:
  - These tools are a work in progress and ideally will be able to yield
  generalize to more than just the legacy EMTF ascii datasets that they
  initially served.

Definitions used in the creation of synthetic mth5 files.

Survey level: 'mth5_path', Path to output h5

Station level: mt_metadata Station() object with station info.
  - the `id` field (name of the station) is required.
  - other station metadata can be added
  - channel_nomenclature
    - The channel_nomenclature was previously stored at the run level.  It makes more sense to store
     this info at the station level, as the only reason the nomenclature would change (that I can
     think of) would be if the acquistion system changed, in which case it would make the most
     sense to initialize a new station object.

Run level: 'columns', :channel names as a list; ["hx", "hy", "hz", "ex", "ey"]
Run level: 'raw_data_path', Path to ascii data source
Run level: 'noise_scalars', dict keyed by channel, default is zero,
Run level: 'nan_indices', iterable of integers, where to put nan [
Run level: 'filters', dict of filters keyed by columns
Run level: 'run_id', name of the run
Run level: 'sample_rate', 1.0


"""

import pathlib
import scipy.signal as ssig
from typing import Dict, List, Optional, Union

import mt_metadata.timeseries
import pandas as pd
from loguru import logger
from mt_metadata.timeseries.filters.helper_functions import make_coefficient_filter
from mt_metadata.timeseries import Run
from mt_metadata.timeseries import Station
from mt_metadata.processing.aurora import ChannelNomenclature
from mt_metadata.processing.aurora.channel_nomenclature import SupportedNomenclatureEnum

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
        sample_rate: float,
        channels: List[str],
        raw_data_path: Optional[Union[str, pathlib.Path]] = None,
        noise_scalars: Optional[dict] = None,
        nan_indices: Optional[dict] = None,
        filters: Optional[dict] = None,
        start: Optional[str] = None,
        timeseries_dataframe: Optional[pd.DataFrame] = None,
        data_source: str = "legacy emtf ascii",
    ) -> None:
        """
        Constructor.

        :param id: label for the run
        :type id: str
        :param sample_rate: sample rate of the time series in Hz.
        :type sample_rate: float
        :param channels: the channel names to include in the run.
        :type channels: List[str]
        :param raw_data_path: Path to ascii data source
        :type raw_data_path: Union[str, pathlib.Path, None]
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
        :type data_source: str

        """
        run_metadata = Run()
        run_metadata.id = id
        run_metadata.sample_rate = sample_rate
        run_metadata.time_period.start = start

        self._timeseries_dataframe = (
            timeseries_dataframe  # normally None for legacy EMTF data
        )
        if isinstance(self._timeseries_dataframe, pd.DataFrame):
            self.data_source = "dataframe"
        else:
            self.data_source = data_source
            self.raw_data_path = raw_data_path

        # set channel names
        self.channels = channels

        # Set scale factors for adding noise to individual channels
        self.noise_scalars = noise_scalars
        if noise_scalars is None:
            self.noise_scalars = {}
            for channel in self.channels:
                self.noise_scalars[channel] = 0.0

        # Set indices for adding nan to individual channels
        if nan_indices is None:
            self.nan_indices = (
                {}
            )  # TODO: make this consistent with noise_scalars, None or empty dict.

        # Set filters individual channels
        if filters is None:
            self.filters = (
                {}
            )  # TODO: make this consistent with noise_scalars, None or empty dict.

        # run_metadata.add_base_attribute("")
        self.run_metadata = run_metadata

    def _get_timeseries_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Returns time series data in a dataframe with columns named for EM field component.

        Up-samples data to self.run_metadata.sample_rate, which is treated as in integer,
        in teh case that self.data_source == "legacy emtf ascii".
        Only tested for 8, to make 8Hz data for testing.  If run.sample_rate is default (1.0)
        then no up-sampling takes place.

        :rtype df: pandas.DataFrame
        :return df: The time series data for the synthetic run

        """
        if isinstance(self._timeseries_dataframe, pd.DataFrame):
            msg = f"Run Data appear to be already set in dataframe"
            logger.info(msg)
            return self._timeseries_dataframe

        elif self.data_source == "legacy emtf ascii":
            ascii_file = LegacyEMTFAsciiFile(file_path=self.raw_data_path)
            df = ascii_file.load_dataframe(
                channel_names=self.channels, sample_rate=self.run_metadata.sample_rate
            )

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
        station_metadata: Station,
        mth5_name: Optional[Union[str, pathlib.Path]] = None,
        channel_nomenclature_keyword: SupportedNomenclatureEnum = SupportedNomenclatureEnum.default,  # TODO: rename to channel_nomenclature_keyword
    ) -> None:
        """
        Constructor.

        :param station_metadata: mt_metadata objeect with station metadata
        :type id: Station
        :param mth5_name: The name of the h5 file to which the station data and metadata will be written.
        :type mth5_name: Optional[Union[str, pathlib.Path]]
        :param channel_nomenclature_keyword: the keyword for the channel nomenclature
        :type channel_nomenclature_keyword: str


        """
        self.station_metadata = station_metadata
        self.runs = []
        self.mth5_name = mth5_name
        self.channel_nomenclature_keyword = channel_nomenclature_keyword
        self._channel_nomenclature = None

        self.station_metadata.channels_recorded = self.channel_nomenclature.channels

    @property
    def channel_nomenclature(self):
        if self._channel_nomenclature is None:
            self._channel_nomenclature = ChannelNomenclature(
                keyword=self.channel_nomenclature_keyword
            )
        return self._channel_nomenclature


def make_station_01(
    channel_nomenclature: SupportedNomenclatureEnum = SupportedNomenclatureEnum.default,
) -> SyntheticStation:
    """
        This method prepares the metadata needed to generate an mth5 with syntheric data.

    :param channel_nomenclature: Must be one of the nomenclatures defined in SupportedNomenclatureEnum
    :type channel_nomenclature: str

    :return: Object with all info needed to generate MTH5 file from synthetic data.
    :rtype: SyntheticStation

    """
    station_metadata = Station()
    station_metadata.id = "test1"
    station_metadata.location.latitude = (
        17.996  # TODO: Add more metadata here as an example
    )

    # initialize SyntheticStation
    station = SyntheticStation(
        station_metadata=station_metadata,
        channel_nomenclature_keyword=channel_nomenclature,  # Needed to assign channel types in RunTS
    )

    station.mth5_name = f"{station_metadata.id}.h5"

    run_001 = SyntheticRun(
        id="001",
        sample_rate=1.0,
        channels=station.channel_nomenclature.channels,
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        start=None,
    )

    # assign indices to set to Nan (not used 2024-06-06)
    nan_indices = {}
    for ch in run_001.channels:
        nan_indices[ch] = []
        if ch == station.channel_nomenclature.hx:
            nan_indices[ch].append([11, 100])
        if ch == station.channel_nomenclature.hy:
            nan_indices[ch].append([11, 100])
            nan_indices[ch].append([20000, 444])
    run_001.nan_indices = nan_indices

    # assign some filters to the channels
    filters = {}
    for ch in run_001.channels:
        if ch in station.channel_nomenclature.ex_ey:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in station.channel_nomenclature.hx_hy_hz:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]
    run_001.filters = filters

    station.runs = [
        run_001,
    ]

    return station


def make_station_02(
    channel_nomenclature: SupportedNomenclatureEnum = SupportedNomenclatureEnum.default,
) -> SyntheticStation:
    """
    Just like station 1, but the data are different

    :param channel_nomenclature: Must be one of the nomenclatures defined in SupportedNomenclatureEnum
    :type channel_nomenclature: SupportedNomenclatureEnum
    :return: Object with all info needed to generate MTH5 file from synthetic data.
    :rtype: SyntheticStation

    """
    test2 = make_station_01(channel_nomenclature=channel_nomenclature)
    test2.station_metadata.id = "test2"
    test2.mth5_name = "test2.h5"
    test2.runs[0].raw_data_path = ASCII_DATA_PATH.joinpath("test2.asc")

    nan_indices = {}
    for channel in test2.runs[0].channels:
        nan_indices[channel] = []
    test2.runs[0].nan_indices = nan_indices

    return test2


def make_station_03(
    channel_nomenclature: SupportedNomenclatureEnum = SupportedNomenclatureEnum.default,
) -> SyntheticStation:
    """
    Create a synthetic station with multiple runs.  Rather than generate fresh
    synthetic data, we just reuse test1.asc for each run.

    :param channel_nomenclature: Literal, Must be one of the nomenclatures defined in "channel_nomenclatures.json"
    :type channel_nomenclature: SupportedNomenclatureEnum
    :rtype: SyntheticStation
    :return: Object with all info needed to generate MTH5 file from synthetic data.

    """

    station_metadata = Station()
    station_metadata.id = "test3"
    station = SyntheticStation(
        station_metadata=station_metadata,
        channel_nomenclature_keyword=channel_nomenclature,
    )
    station.mth5_name = "test3.h5"

    channels = station.channel_nomenclature.channels

    nan_indices = {}
    for ch in channels:
        nan_indices[ch] = []

    filters = {}
    for ch in channels:
        if ch in station.channel_nomenclature.ex_ey:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in station.channel_nomenclature.hx_hy_hz:
            filters[ch] = [FILTERS["10x"].name, FILTERS["0.1x"].name]

    run_001 = SyntheticRun(
        id="001",
        sample_rate=1.0,
        channels=channels,
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        nan_indices=nan_indices,
        filters=filters,
        start="1980-01-01T00:00:00+00:00",
    )

    noise_scalars = {}
    for ch in channels:
        noise_scalars[ch] = 2.0
    run_002 = SyntheticRun(
        id="002",
        sample_rate=1.0,
        channels=channels,
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
        start="1980-01-02T00:00:00+00:00",
    )

    for ch in channels:
        noise_scalars[ch] = 5.0
    run_003 = SyntheticRun(
        id="003",
        sample_rate=1.0,
        channels=channels,
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
        start="1980-01-03T00:00:00+00:00",
    )

    for ch in channels:
        noise_scalars[ch] = 10.0
    run_004 = SyntheticRun(
        id="004",
        sample_rate=1.0,
        channels=channels,
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        noise_scalars=noise_scalars,
        nan_indices=nan_indices,
        filters=filters,
        start="1980-01-04T00:00:00+00:00",
    )

    run_001.filters = filters
    run_002.filters = filters
    run_003.filters = filters
    run_004.filters = filters

    station.runs = [run_001, run_002, run_003, run_004]

    return station


def make_station_04(
    channel_nomenclature: SupportedNomenclatureEnum = SupportedNomenclatureEnum.default,
) -> SyntheticStation:
    """
    Just like station 01, but data are resampled to 8Hz

    :param channel_nomenclature: Literal, Must be one of the nomenclatures defined in "channel_nomenclatures.json"
    :type channel_nomenclature: SupportedNomenclatureEnum
    :rtype: SyntheticStation
    :return: Object with all info needed to generate MTH5 file from synthetic data.
    """

    station_metadata = Station()
    station_metadata.id = "test1"

    station = SyntheticStation(
        station_metadata=station_metadata,
        channel_nomenclature_keyword=channel_nomenclature,
    )
    station.mth5_name = "test_04_8Hz.h5"

    run_001 = SyntheticRun(
        id="001",
        sample_rate=8.0,
        channels=station.channel_nomenclature.channels,
        raw_data_path=ASCII_DATA_PATH.joinpath("test1.asc"),
        start=None,
    )
    run_001.nan_indices = {}

    filters = {}
    for ch in run_001.channels:
        if ch in station.channel_nomenclature.ex_ey:
            filters[ch] = [
                FILTERS["1x"].name,
            ]
        elif ch in station.channel_nomenclature.hx_hy_hz:
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


class LegacyEMTFAsciiFile:
    """
    This class can be used to interact with the legacy synthetic data files
    that were originally in EMTF.

    Development Notes:
     As of 2025-02-03 the only LegacyEMTFAsciiFile date sources are sampled at 1Hz.
     One-off upsampling can be handled in this class if the requested sample rate differs.

    """

    IMPLICIT_SAMPLE_RATE = 1.0  # Hz

    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path

    def load_dataframe(
        self,
        channel_names: list,
        sample_rate: float,
    ) -> pd.DataFrame:
        """
        Loads an EMTF legacy ASCII time series into a dataframe.

        These files have an awkward whitespace separator, and also need to have the
         electric field channels inverted to fix a phase swap.

        :param channel_names: The names of the channels in the legacy EMTF file, in order.
        :type channel_names: list
        :param sample_rate: The sample rate of the output time series in Hz.
        :type sample_rate: float

        :return df: The labelled time series from the legacy EMTF file.
        :rtype df: pd.DataFrame

        """

        # read in data
        df = pd.read_csv(self.file_path, names=channel_names, sep="\s+")

        # Invert electric channels to fix phase swap due to modeling coordinates.
        # Column indices are used to avoid handling channel nomenclature here.
        df[df.columns[-2]] = -df[df.columns[-2]]  # df["ex"] = -df["ex"]
        df[df.columns[-1]] = -df[df.columns[-1]]  # df["ey"] = -df["ey"]

        # Temporary kludge: One-off handling for a test case to upsample data.
        # TODO: delete this once synthetic data module is built can offer multiple sample rates
        if sample_rate != self.IMPLICIT_SAMPLE_RATE:
            df_orig = df.copy(deep=True)
            new_data_dict = {}
            for ch in df.columns:
                data = df_orig[ch].to_numpy()
                new_data_dict[ch] = ssig.resample(data, int(sample_rate) * len(df_orig))
            df = pd.DataFrame(data=new_data_dict)

        return df


def main():
    # sr = SyntheticRun("001")
    make_station_04()


if __name__ == "__main__":
    main()
