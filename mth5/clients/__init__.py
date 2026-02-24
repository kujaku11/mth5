# package file

from .fdsn import FDSN
from .geomag import USGSGeomag
from .phoenix import PhoenixClient
from .zen import ZenClient
from .lemi import LEMIClient, LEMI424Client  # LEMI424Client is deprecated alias
from .metronix import MetronixClient
from .nims import NIMSClient
from .uoa import UoAClient
from .make_mth5 import MakeMTH5

__all__ = [
    "FDSN",
    "USGSGeomag",
    "PhoenixClient",
    "ZenClient",
    "LEMIClient",
    "LEMI424Client",  # Deprecated alias for backward compatibility
    "MetronixClient",
    "NIMSClient",
    "UoAClient",
    "MakeMTH5",
]
