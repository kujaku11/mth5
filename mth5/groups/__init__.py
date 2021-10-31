"""
Import all Group objects
"""
## !!! DO NOT CHANGE ORDER !!!
from .base import BaseGroup
from .reports import ReportsGroup
from .standards import StandardsGroup
from .filters import FiltersGroup
from .master_station_run_channel import (
    MasterStationGroup,
    StationGroup,
    RunGroup,
    ElectricDataset,
    MagneticDataset,
    ChannelDataset,
    AuxiliaryDataset,
)
from .survey import MasterSurveyGroup, SurveyGroup


__all__ = [
    "BaseGroup",
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
]
