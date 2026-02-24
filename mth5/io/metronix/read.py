from mth5.io.metronix.metronix_atss import read_atss, ATSS
from loguru import logger
import pathlib

def read_metronix(file_path: str | pathlib.Path) -> ATSS:
    """
    Read a Metronix ATSS file and return an ATSS object.

    Parameters
    ----------
    file_path : str or Path
        Path to the Metronix ATSS file to read.

    Returns
    -------
    ATSS
        An ATSS object containing the data from the file.
    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    if file_path.suffix.lower() not in [".atss", ".ats"]:
        raise ValueError(f"File {file_path} does not have a valid Metronix extension (.atss or .ats)")
    
    if file_path.suffix.lower() == ".ats":
        logger.warning(f"File {file_path} has .ats extension, but .atss is expected. Attempting to read as ATSS.")
        raise NotImplementedError("Reading .ats files is not implemented yet. Please convert to .atss format.")
    elif file_path.suffix.lower() == ".atss":
        logger.info(f"Reading Metronix ATSS file: {file_path}")
        readed = read_atss(file_path)
        return readed
