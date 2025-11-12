"""
Convert MTH5 to other formats

- MTH5 -> miniSEED + StationXML
"""

from __future__ import annotations

import datetime

# ==================================================================================
# Imports
# ==================================================================================
import re
from pathlib import Path
from typing import Any

from loguru import logger
from mt_metadata.timeseries.stationxml import XMLInventoryMTExperiment
from obspy import read
from obspy.core import UTCDateTime

from mth5.mth5 import MTH5


# ==================================================================================


class MTH5ToMiniSEEDStationXML:
    """
    Convert MTH5 files to miniSEED and StationXML formats.

    This class provides functionality to convert magnetotelluric data stored
    in MTH5 format to industry-standard miniSEED time series files and
    StationXML metadata files for data exchange and archival purposes.

    Parameters
    ----------
    mth5_path : str, Path, or None, default None
        Path to the input MTH5 file to be converted
    save_path : str, Path, or None, default None
        Directory path where output files will be saved. If None, uses
        the parent directory of mth5_path
    network_code : str, default "ZU"
        Two-character FDSN network code for the output files
    use_runs_with_data_only : bool, default True
        If True, only process runs that contain actual time series data
    **kwargs : dict
        Additional keyword arguments to set as instance attributes

    Attributes
    ----------
    mth5_path : Path or None
        Path to the MTH5 input file
    save_path : Path
        Directory where output files are saved
    network_code : str
        FDSN network code for output files
    use_runs_with_data_only : bool
        Flag to process only runs with data
    encoding : str or None
        Encoding format for miniSEED files

    Examples
    --------
    >>> converter = MTH5ToMiniSEEDStationXML(
    ...     mth5_path="/path/to/data.h5",
    ...     network_code="MT",
    ...     save_path="/path/to/output"
    ... )
    >>> xml_file, mseed_files = converter.convert_mth5_to_ms_stationxml()
    """

    def __init__(
        self,
        mth5_path: str | Path | None = None,
        save_path: str | Path | None = None,
        network_code: str = "ZU",
        use_runs_with_data_only: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MTH5 to miniSEED/StationXML converter.

        Parameters
        ----------
        mth5_path : str, Path, or None, default None
            Path to the input MTH5 file to be converted
        save_path : str, Path, or None, default None
            Directory path where output files will be saved. If None, uses
            the parent directory of mth5_path
        network_code : str, default "ZU"
            Two-character FDSN network code for the output files
        use_runs_with_data_only : bool, default True
            If True, only process runs that contain actual time series data
        **kwargs : dict
            Additional keyword arguments to set as instance attributes
        """
        self._network_code_pattern = r"^[a-zA-Z0-9]{2}$"
        self.mth5_path = mth5_path
        self.save_path = save_path
        self.network_code = network_code
        self.use_runs_with_data_only = use_runs_with_data_only
        self.encoding = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def mth5_path(self) -> Path | None:
        """
        Path to the MTH5 input file.

        Returns
        -------
        Path or None
            Path to the MTH5 file to be converted, or None if not set.
        """
        return self._mth5_path

    @mth5_path.setter
    def mth5_path(self, value: str | Path | None) -> None:
        """
        Set the MTH5 file path with validation.

        Parameters
        ----------
        value : str, Path, or None
            Path to the MTH5 file. Must exist if not None.

        Raises
        ------
        TypeError
            If value cannot be converted to a Path object.
        FileNotFoundError
            If the specified file does not exist.
        """
        if value is None:
            self._mth5_path = None
            return
        try:
            value = Path(value)
        except Exception as error:
            raise TypeError(f"Could not convert value to Path: {error}")

        if not value.exists():
            raise FileExistsError(f"Could not find {value}")

        self._mth5_path = value

    @property
    def save_path(self) -> Path:
        """
        Directory path where output files will be saved.

        Returns
        -------
        Path
            Directory path for saving miniSEED and StationXML files.
        """
        return self._save_path

    @save_path.setter
    def save_path(self, value: str | Path | None) -> None:
        """
        Set the save directory path with automatic creation.

        Parameters
        ----------
        value : str, Path, or None
            Directory path where files will be saved. If None, uses the
            parent directory of mth5_path or current working directory.

        Notes
        -----
        Creates the directory if it doesn't exist.
        """
        """Set the save path, if None set to parent directory of mth5_path"""
        if value is None:
            if self._mth5_path is None:
                self._save_path = Path().cwd()
            else:
                self._save_path = self._mth5_path.parent
        else:
            self._save_path = Path(value)

        if not self._save_path.exists():
            self._save_path.mkdir(exists_ok=True)

    @property
    def network_code(self) -> str:
        """
        Two-character FDSN network code.

        Returns
        -------
        str
            Alphanumeric string of exactly 2 characters as required by FDSN DMC.
        """
        return self._network_code

    @network_code.setter
    def network_code(self, value: str) -> None:
        """
        Set the FDSN network code with validation.

        Parameters
        ----------
        value : str
            Two-character alphanumeric network code.

        Raises
        ------
        ValueError
            If value is None or doesn't match the required 2-character pattern.

        Notes
        -----
        Request temporary codes from https://www.fdsn.org/networks/request/temp/
        """
        if value is None:
            raise ValueError(
                "Must input a network code.  "
                "Request a temporary code from https://www.fdsn.org/networks/request/temp/"
            )
        if not re.match(self._network_code_pattern, value):
            raise ValueError(
                f"{value} is not a valid network code. It must be 2 alphanumeric characters"
            )
        self._network_code = value

    @classmethod
    def convert_mth5_to_ms_stationxml(
        cls,
        mth5_path: str | Path,
        save_path: str | Path | None = None,
        network_code: str = "ZU",
        use_runs_with_data_only: bool = True,
        **kwargs: Any,
    ) -> tuple[Path, list[Path]]:
        """
        Convert an MTH5 file to miniSEED and StationXML formats.

        Class method that provides a convenient interface to convert MTH5 data
        to standard seismological formats for data exchange and archival.

        Parameters
        ----------
        mth5_path : str or Path
            Path to the input MTH5 file to be converted
        save_path : str, Path, or None, default None
            Directory where output files will be saved. If None, uses the
            parent directory of mth5_path
        network_code : str, default "ZU"
            Two-character FDSN network code for output files
        use_runs_with_data_only : bool, default True
            If True, only process runs containing actual time series data
        **kwargs : dict
            Additional keyword arguments passed to converter initialization

        Returns
        -------
        tuple[Path, list[Path]]
            Tuple containing:
            - Path to the generated StationXML file
            - List of paths to generated miniSEED files (one per day per channel)

        Examples
        --------
        >>> xml_file, mseed_files = MTH5ToMiniSEEDStationXML.convert_mth5_to_ms_stationxml(
        ...     "/path/to/data.h5",
        ...     network_code="MT",
        ...     save_path="/output/directory"
        ... )
        >>> print(f"Created {len(mseed_files)} miniSEED files and {xml_file}")
        """

        converter = cls(
            mth5_path=mth5_path,
            save_path=save_path,
            network_code=network_code,
            use_runs_with_data_only=use_runs_with_data_only,
            **kwargs,
        )

        with MTH5() as m:
            m.open_mth5(converter.mth5_path)
            experiment = m.to_experiment(has_data=converter.use_runs_with_data_only)
            stream_list = []
            for row in m.run_summary.itertuples():
                if row.has_data:
                    run_ts = m.from_reference(row.run_hdf5_reference).to_runts()
                    if converter.encoding is None:
                        encoding = get_encoding(run_ts)
                    else:
                        encoding = converter.encoding
                    stream = run_ts.to_obspy_stream(
                        network_code=converter.network_code, encoding=encoding
                    )
                    # write to miniseed files
                    stream_list += converter.split_ms_to_days(
                        stream, converter.save_path, encoding
                    )

        # write StationXML
        experiment.surveys[0].fdsn.network = converter.network_code

        translator = XMLInventoryMTExperiment()
        xml_fn = converter.save_path.joinpath(f"{converter.mth5_path.stem}.xml")
        stationxml = translator.mt_to_xml(
            experiment,
            stationxml_fn=xml_fn,
        )
        logger.info(f"Wrote StationXML to {xml_fn}")

        return xml_fn, stream_list

    def split_ms_to_days(self, streams, save_path: Path, encoding: str) -> list[Path]:
        """
        Split miniSEED traces into daily files.

        Splits continuous time series traces into separate files for each day
        to conform with standard seismological data archiving practices.

        Parameters
        ----------
        streams : obspy.Stream
            Stream object containing traces to be split by day
        save_path : Path
            Directory where daily miniSEED files will be saved
        encoding : str
            Data encoding format for miniSEED files (e.g., 'INT32', 'FLOAT64')

        Returns
        -------
        list[Path]
            List of paths to the generated daily miniSEED files

        Notes
        -----
        Files are named using the pattern:
        {network}_{station}_{location}_{channel}_{YYYY_MM_DDTHH_MM_SS}.mseed
        """
        fn_list = []
        for tr in streams:
            start_time = tr.stats.starttime
            end_time = tr.stats.endtime

            # Split the trace by day
            current_time = start_time
            while current_time < end_time:
                next_day = UTCDateTime(current_time.date + datetime.timedelta(days=1))
                if next_day > end_time:
                    next_day = end_time

                # Slice the trace for the current day
                tr_day = tr.slice(current_time, next_day)

                # Generate the output file name
                output_file = save_path.joinpath(
                    f"{tr.stats.network}_{tr.stats.station}_{tr.stats.location}_{tr.stats.channel}_{current_time.isoformat().replace('-', '_').replace(':', '_')}.mseed"
                )
                logger.info(f"Wrote miniseed file to: {output_file}")

                fn_list.append(output_file)

                # Write the sliced trace to a new MiniSEED file
                tr_day.write(output_file, format="MSEED", reclen=256, encoding=encoding)

                # Move to the next day
                current_time = next_day

        return fn_list


def get_encoding(run_ts) -> str:
    """
    Determine consistent data encoding for miniSEED files across channels.

    Analyzes data types across all channels in a run and selects a median
    encoding to ensure compatibility in miniSEED file generation.

    Parameters
    ----------
    run_ts : RunTS
        Run time series object containing multiple channels of data

    Returns
    -------
    str
        String identifier for miniSEED encoding format (e.g., 'INT32', 'FLOAT64')

    Notes
    -----
    Uses median data type to handle mixed precision datasets. Automatically
    converts INT64 to INT32 for miniSEED compatibility since some readers
    don't support 64-bit integers.

    Examples
    --------
    >>> encoding = get_encoding(run_timeseries)
    >>> print(f"Selected encoding: {encoding}")
    """
    dtypes = [run_ts.dataset[ch].data.dtype.name for ch in run_ts.channels]
    encoding = sorted(dtypes)[int(len(dtypes) / 2)].upper()
    if encoding in ["INT64"]:
        encoding = "INT32"
        logger.warning("Casting INT64 to INT32")

    return encoding


def split_miniseed_by_day(input_file: str | Path) -> list[Path]:
    """
    Split an existing miniSEED file into daily files.

    Utility function to split a multi-day miniSEED file into separate files
    for each calendar day, following standard seismological archiving practices.

    Parameters
    ----------
    input_file : str or Path
        Path to the input miniSEED file to be split

    Returns
    -------
    list[Path]
        List of paths to the generated daily miniSEED files

    Notes
    -----
    Output files are named using the pattern:
    {network}.{station}.{location}.{channel}.{YYYY-MM-DD}.mseed

    Files are saved in the same directory as the input file.

    Examples
    --------
    >>> daily_files = split_miniseed_by_day("/path/to/continuous.mseed")
    >>> print(f"Created {len(daily_files)} daily files")
    """
    save_path = Path(input_file).parent
    # Read the MiniSEED file
    st = read(input_file)

    tr_list = []
    # Iterate over each trace in the stream
    for tr in st:
        start_time = tr.stats.starttime
        end_time = tr.stats.endtime

        # Split the trace by day
        current_time = start_time
        while current_time < end_time:
            next_day = UTCDateTime(current_time.date + datetime.timedelta(days=1))
            if next_day > end_time:
                next_day = end_time

            # Slice the trace for the current day
            tr_day = tr.slice(current_time, next_day)

            # Generate the output file name
            output_file = save_path.joinpath(
                f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}.{current_time.date}.mseed"
            )

            # Write the sliced trace to a new MiniSEED file
            tr_day.write(output_file, format="MSEED")
            tr_list.append(output_file)

            # Move to the next day
            current_time = next_day

    return tr_list
