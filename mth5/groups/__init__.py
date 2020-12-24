"""
Import all Group objects
"""
## !!! DO NOT CHANGE ORDER !!!
from .base import BaseGroup
from .reports import ReportsGroup
from .standards import StandardsGroup
from .filters import FiltersGroup
from .master_station_run_channel import MasterStationGroup, StationGroup, RunGroup, ElectricDataset, MagneticDataset, ChannelDataset, AuxiliaryDataset
from .survey import SurveyGroup
from .filter_dataset import FilterDataset



__all__ = ["BaseGroup", "SurveyGroup", "ReportsGroup", "StandardsGroup", "StationGroup",
            "MasterStationGroup", "FiltersGroup", "RunGroup", "ChannelDataset",  
            "ElectricDataset", "MagneticDataset", "AuxiliaryDataset", "FilterDataset"]