"""
Import all Group objects
"""
## !!! DO NOT CHANGE ORDER !!!
from .base import BaseGroup
from .reports import ReportsGroup
from .standards import StandardsGroup
from .filters import FiltersGroup
from .estimate_dataset import EstimateDataset
from .fc_dataset import FCChannelDataset
from .fourier_coefficients import MasterFCGroup, FCGroup, FCDecimationGroup
from .transfer_function import TransferFunctionsGroup, TransferFunctionGroup
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
    "FiltersGroup",
    "RunGroup",
    "ChannelDataset",
    "ElectricDataset",
    "MagneticDataset",
    "AuxiliaryDataset",
    "EstimateDataset",
    "FCChannelDataset",
    "TransferFunctionGroup",
    "TransferFunctionsGroup",
    "MasterFCGroup",
    "FCGroup",
    "FCDecimationGroup",
    "ExperimentGroup",
    "SurveyGroup",
    "ReportsGroup",
    "StandardsGroup",
    "StationGroup",
    "MasterStationGroup",
    "MasterSurveyGroup",
]
