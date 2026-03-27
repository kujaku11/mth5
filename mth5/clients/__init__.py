# package file

from .fdsn import FDSN
from .geomag import USGSGeomag
from .phoenix import PhoenixClient
from .zen import ZenClient
from .lemi import LEMIClient, LEMI424Client  # LEMI424Client is deprecated alias
from .metronix import MetronixClient
<<<<<<< HEAD
from .uoa import UoAClient
=======
from .nims import NIMSClient
>>>>>>> master
from .make_mth5 import MakeMTH5

__all__ = [
    "FDSN",
    "USGSGeomag",
    "PhoenixClient",
    "ZenClient",
    "LEMIClient",
    "LEMI424Client",  # Deprecated alias for backward compatibility
    "MetronixClient",
<<<<<<< HEAD
    "UoAClient",
=======
    "NIMSClient",
>>>>>>> master
    "MakeMTH5",
]
