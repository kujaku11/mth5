"""
Import all Group objects
"""
## !!! DO NOT CHANGE ORDER !!!
from .base import BaseGroup
from .reports import ReportsGroup
from .standards import StandardsGroup
from .filters import FiltersGroup
from .estimate_dataset import EstimateDataset
from .fourier_coefficients import MasterFCGroup, FCGroup, FCChannel
from .transfer_function import TransferFunctionGroup
from .channel_dataset import (
    ElectricDataset,
    MagneticDataset,
    ChannelDataset,
    AuxiliaryDataset,
)
from .run import RunGroup
from .station import MasterStationGroup, StationGroup
from .survey import MasterSurveyGroup, SurveyGroup
from .experiment import ExperimentGroup


__all__ = [
    "BaseGroup",
    "ExperimentGroup",
    "SurveyGroup",
    "ReportsGroup",
    "StandardsGroup",
    "StationGroup",
    "MasterStationGroup",
    "MasterSurveyGroup",
    "FiltersGroup",
    "RunGroup",
    "ChannelDataset",
    "ElectricDataset",
    "MagneticDataset",
    "AuxiliaryDataset",
    "EstimateDataset",
    "TransferFunctionGroup",
    "MasterFCGroup",
    "FCGroup",
    "FCChannel",
]
