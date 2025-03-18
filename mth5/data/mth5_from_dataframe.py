"""
    This module will contain example code for ingesting dataframes into mth5.
     - The definition of mth5 follows the example in mth5.data.make_mth5_from_ascii.py


"""
import numpy as np
import pathlib

import pandas as pd
from mt_metadata.timeseries import Station
from mt_metadata.transfer_functions.processing.aurora.channel_nomenclature import SupportedNomenclature
from mth5.data.station_config import SyntheticRun, SyntheticStation
from mth5.timeseries.spectre import MultivariateDataset
from mth5.timeseries.spectre.multiple_station import MultivariateLabelScheme

MTH5_PATH = pathlib.Path(".")

STATION_IDS = [f"site_0{x}" for x in [1, 2]]


def make_synthetic_data_config_dict():
    """
        One off dict with some controls for synthetic data properties

        :return:
    """
    pass


def make_synthetic_time_series():
    """
        One off method to generate some synthetic data.

    :return:

    """

    # Set some paramters
    n_observations = 5000

    # define the spread of the magnetic field amplitudes (in nanoTesla)
    std_deviations = {}
    std_deviations["site_01"] = {}
    std_deviations["site_02"] = {}
    std_deviations["site_01"]["B"] = {}
    std_deviations["site_02"]["B"] = {}
    std_deviations["site_01"]["B"]["x"] = 100.0  # nT
    std_deviations["site_01"]["B"]["y"] = 100.0  # nT
    std_deviations["site_02"]["B"]["x"] = 100.0  # nT
    std_deviations["site_02"]["B"]["y"] = 100.0  # nT

    df_site_01_dict = {}
    df_site_01_dict["bx"] = np.random.randn(n_observations)
    df_site_01_dict["by"] = np.random.randn(n_observations)
    df_site_01_dict["ex"] = np.random.randn(n_observations)
    df_site_01_dict["ey"] = np.random.randn(n_observations)
    df_site_01 = pd.DataFrame(data=df_site_01_dict)

    df_site_02_dict = {}
    df_site_02_dict["bx"] = np.random.randn(n_observations)
    df_site_02_dict["by"] = np.random.randn(n_observations)
    df_site_02_dict["ex"] = np.random.randn(n_observations)
    df_site_02_dict["ey"] = np.random.randn(n_observations)
    df_site_02 = pd.DataFrame(data=df_site_02_dict)

    return df_site_01, df_site_02

df_site_01, df_site_02 = make_synthetic_time_series()

xr_site_01 = df_site_01.to_xarray()
label_scheme = MultivariateLabelScheme(join_char="__")

output = MultivariateDataset(dataset=xr_site_01, label_scheme=label_scheme)
print("OK")

# def make_site_01(channel_nomenclature: SupportedNomenclature = "musgraves") -> SyntheticStation:
#     """
#         This method prepares the metadata needed to generate an mth5 with synthetic data.
#
#     :param channel_nomenclature: Must be one of the nomenclatures defined in SupportedNomenclature
#     :type channel_nomenclature: str
#
#     :return: Object with all info needed to generate MTH5 file from synthetic data.
#     :rtype: SyntheticStation
#
#     """
#     df_site_01, df_site_02 = make_synthetic_time_series()
#
#     station_metadata = Station()
#     station_metadata.id = "Site_01"
#     station_metadata.location.latitude = 17.996  # TODO: Add more metadata here as an example
#
#     # initialize SyntheticStation
#     site_01 = SyntheticStation(
#         station_metadata=station_metadata,
#         channel_nomenclature_keyword=channel_nomenclature  # Needed to assign channel types in RunTS
#     )
#
#     site_01.mth5_name = f"{station_metadata.id}.h5"
#
#     run_001 = SyntheticRun(
#         id="001",
#         sample_rate=1.0,
#         channels = station.channel_nomenclature.channels,
#         timeseries_dataframe = df_site_01,  #  TODO: this df doesnt exist yet
#         start=None,
#     )
#
#     # assign some filters to the channels
#     # filters = {}
#     # for ch in run_001.channels:
#     #     if ch in station.channel_nomenclature.ex_ey:
#     #         filters[ch] = [
#     #             FILTERS["1x"].name,
#     #         ]
#     #     elif ch in station.channel_nomenclature.hx_hy_hz:
#     #         filters[ch] = [
#     #             FILTERS["10x"].name,
#     #             FILTERS["0.1x"].name
#     #         ]
#     # run_001.filters = filters
#
#     station.runs = [
#         run_001,
#     ]
#
#     return station
