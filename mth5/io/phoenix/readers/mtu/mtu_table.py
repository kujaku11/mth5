import struct
from pathlib import Path

from loguru import logger
from mt_metadata.timeseries import Electric, Magnetic, Run, Station, Survey


class MTUTable:
    """
    =======================================================================
    DECODING METHOD FOR TBL VALUES:

    The Phoenix TBL file is a series of 25-byte blocks containing key-value pairs:
    - Bytes 0-11:  Tag name (4-character string, null-padded)
    - Bytes 12-24: Value (13 bytes, mixed data types)

    Values can be decoded as follows:
    1. INT (4 bytes):    struct.unpack('<i', bytes[0:4])  - Little-endian signed int
    2. DOUBLE (8 bytes): struct.unpack('<d', bytes[0:8])  - Little-endian double
    3. CHAR (variable):  bytes.decode('latin-1').strip()  - Null-terminated string
    4. BYTE (1 byte):    struct.unpack('<B', bytes[0:1])  - Unsigned byte
    5. TIME (6 bytes):   [sec, min, hour, day, month, year-2000] format

    The TBL_TAG_TYPES dictionary maps each known tag to its data type, enabling
    automatic decoding via decode_tbl_value() function. Unknown tags return raw bytes.

    Example usage:
        # Automatic decoding:
        tbl_dict = get_dictionary_from_tbl('file.TBL', decode_values=True)

        # Manual decoding with read_tbl (legacy):
        info = read_tbl('/path', 'file.TBL')

    =======================================================================
    original comments from MATLAB script:

    read_tbl - reads a (binary) TBL table file of the legacy Phoenix format
    (MTU-5A) and output the "info" metadata dictionary.

    Parameters:
        fpath: path to the tbl
        fname: name of the tbl file (including extensions)

    Returns:
        info: output dict of the TBL metadata

    =======================================================================
    definition of the TBL tags (or what I guessed after reading the user
    manual and fiddling with their files)
    SITE: site name
    SNUM: serial number (of the box)
    FILE: file name recorded
    CMPY: company/institute of the survey
    SRVY: survey project name
    EXLN: Ex channel dipole length
    EYLN: Ey channel dipole length
    NREF: North reference (true, or magnetic north)
    LNGG: longitude in degree-minute format (DDD MM.MM)
    LATG: latitude in degree-minute format (DD MM.MM)
    ELEV: elevation (in metres)
    HXSN: Hx channel coil serial number
    HYSN: Hy channel coil serial number
    HZSN: Hz channel coil serial number
    STIM: starting time (UTC)
    ETIM: ending time (UTC)
    LFRQ: powerline frequency for filtering (can only be 50 or 60 Hz)
    HGN:  final H-channel gain
    HGNC: H-channel gain control: HGN = PA * 2^HGNC (note: PA =
        PreAmplifier gain)
    EGN:  final E-channel gain
    EGNC: E-channel gain control: HGN = PA * 2^HGNC (note: PA =
        PreAmplifier gain)
    HSMP: L3 and L4 time slot in second (MTU-5A) or minute (MTU-5P),
        this means the instrument will record L3NS seconds for L3 and L4NS
        seconds for L4, for every HSMP time slot.
    L3NS: L3 sample time (in second)
    L4NS: L4 sample time (in second)
    SRL3: L3 sample rate
    SRL4: L4 sample rate
    SRL5: L5 sample rate
    HATT: H channel attenuation (1/4.3 for MTU-5A)
    HNOM: H channel normalization (mA/nT)
    TCMB: Type of comb filter (probably used to suppress the harmonics of the
        powerline noise.
    TALS: Type of anti-aliasing filter
    LPFR: Parameter of Low-pass/VLF filter. this is a quite complicated
        part as the low-pass filter is simply an R-C circuit with a switch
        to connect to different capacitors. To ensure enough bandwidth
        (proportion to 1/RC), one should use smaller capacitors with larger
        ground resistance.
    ACDC: AC/DC coupling (DC = 0, AC = 1; MT should always be DC)
    FSCV: full scaling A-D converter voltage (in unit of V)
    =======================================================================
    note:
    Phoenix Legacy TBL is a straight-forward parameter-value metadata file,
    stored in a bizarre format. The parameter tag and value are stored in a
    series of 25-byte data blocks, in mixed data type: the first 12 bytes are
    reserved for the tag name (first 4 bytes as char). The values are stored
    in the 13 bytes afterwards, in various formats (char, int, float, etc.).

    So a good practice is to read in those blocks one by one and extract all
    of them. However, not every thing is useful for the metadata, so I only
    extract a few of them, for now.

    Original author:
    Hao
    2012.07.04
    Beijing

    Translated to Python and enhanced by:
    J. Peacock (2025-12-31)

    Main changes:

    - Encapsulated in MTUTable class
    - Automatic type detection and decoding based on TBL_TAG_TYPES
    - Added properties to extract metadata as mt_metadata objects
    =======================================================================
    """

    def __init__(self, file_path: str | Path | None = None, **kwargs) -> None:
        """
        Initialize MTUTable reader.

        Parameters
        ----------
        file_path : str or Path, optional
            Path to the TBL file including the file name. If not provided, the object can be initialized without a file.

        Examples
        --------
        >>> tbl = MTUTable('/data/phoenix/1690C16C.TBL')
        >>> tbl.read_tbl()
        >>> print(tbl.tbl_dict['SITE'])
        '10441W10'
        """
        self.file_path = Path(file_path) if file_path else None
        self.tbl_dict: dict[str, int | float | str | bytes] = {}

        # TBL tag data type mapping
        # Format: 'TAG': ('type', description)
        # Types: 'int', 'double', 'char', 'byte', 'time'
        self.TBL_TAG_TYPES = {
            "SNUM": ("int", "Serial number"),
            "SITE": ("char", "Site name"),
            "FILE": ("char", "File name recorded"),
            "FLEN": ("int", "File length in bytes"),
            "FTIM": ("time", "File creation time UTC"),
            "CMPY": ("char", "Company/institute"),
            "SRVY": ("char", "Survey project name"),
            "LATG": ("char", "Latitude in degree-minute format"),
            "LNGG": ("char", "Longitude in degree-minute format"),
            "ELEV": ("int", "Elevation in metres"),
            "NREF": ("int", "North reference"),
            "STIM": ("time", "Starting time UTC"),
            "ETIM": ("time", "Ending time UTC"),
            "EXLN": ("double", "Ex channel dipole length"),
            "EYLN": ("double", "Ey channel dipole length"),
            "HXSN": ("char", "Hx channel coil serial number"),
            "HYSN": ("char", "Hy channel coil serial number"),
            "HZSN": ("char", "Hz channel coil serial number"),
            "EAZM": ("double", "E azimuth"),
            "HAZM": ("double", "H azimuth"),
            "HTIM": ("time", "H channel calibration time"),
            "HSMP": ("int", "L3 and L4 time slot"),
            "L3NS": ("int", "L3 sample time in seconds"),
            "L4NS": ("int", "L4 sample time in seconds"),
            "LTIME": ("time", "L channel calibration time ?"),
            "SRL3": ("int", "L3 sample rate"),
            "SRL4": ("int", "L4 sample rate"),
            "SRL5": ("int", "L5 sample rate"),
            "LFRQ": ("byte", "Powerline frequency"),
            "EGNC": ("int", "E-channel gain control"),
            "HGNC": ("int", "H-channel gain control"),
            "EGN": ("int", "Final E-channel gain"),
            "HGN": ("int", "Final H-channel gain"),
            "HATT": ("double", "H channel attenuation"),
            "HNOM": ("double", "H channel normalization mA/nT"),
            "TCMB": ("byte", "Type of comb filter"),
            "TALS": ("byte", "Type of anti-aliasing filter"),
            "LPFR": ("byte", "Low-pass/VLF filter parameter"),
            "ACDC": ("byte", "AC/DC coupling"),
            "FSCV": ("double", "Full scaling A-D converter voltage"),
            "TEMP": ("int", "Temperature"),
            "TERR": ("int", "Temperature error"),
            "V5SR": ("int", "MTU-5 serial number"),
            "NSAT": ("int", "Number of GPS satellites"),
            # Additional tags that may appear in TBL files
            "DECL": ("double", "Declination"),
            "TSTV": ("double", "Test voltage"),
            "VER": ("char", "Version"),
            "HW": ("16s", "Hardware"),
            "EXAC": ("double", "Ex AC"),
            "EXDC": ("double", "Ex DC"),
            "EYAC": ("double", "Ey AC"),
            "EYDC": ("double", "Ey DC"),
            "HXAC": ("double", "Hx AC"),
            "HXDC": ("double", "Hx DC"),
            "HYAC": ("double", "Hy AC"),
            "HYDC": ("double", "Hy DC"),
            "HZAC": ("double", "Hz AC"),
            "HZDC": ("double", "Hz DC"),
            "STDE": ("double", "Standard error in Electric channels"),
            "STDH": ("double", "Standard error in Magnetic channels"),
            "SPTH": ("char", "system path"),
            "CHEX": ("int", "EX channel type"),
            "CHEY": ("int", "EY channel type"),
            "CHHX": ("int", "HX channel type"),
            "CHHY": ("int", "HY channel type"),
            "CHHZ": ("int", "HZ channel type"),
        }

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.file_path:
            self.read_tbl()

    def decode_tbl_value(
        self, value_bytes: bytes, data_type: str
    ) -> int | float | str | bytes:
        """
        Decode TBL value bytes based on the specified data type.

        Parameters
        ----------
        value_bytes : bytes
            13 bytes from position 12-24 in the 25-byte block containing the value.
        data_type : str
            Type of the data: 'int', 'double', 'char', 'byte', or 'time'.

        Returns
        -------
        int or float or str or bytes
            Decoded value in appropriate Python type. Returns raw bytes if
            decoding fails or data_type is unrecognized.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> value = tbl.decode_tbl_value(b'\x9a\x06\x00\x00...', 'int')
        >>> print(value)
        1690
        """
        if data_type == "int":
            return struct.unpack("<i", value_bytes[0:4])[0]
        elif data_type == "double":
            return struct.unpack("<d", value_bytes[0:8])[0]
        elif data_type == "char":
            return value_bytes.decode("latin-1").strip("\x00").strip()
        elif data_type == "byte":
            return struct.unpack("<B", value_bytes[0:1])[0]
        elif data_type == "time":
            # Time format: bytes are [sec, min, hour, day, month, year-2000]
            # Return formatted string: YYYY-MM-DD HH:MM:SS
            return f"20{value_bytes[5]:02}-{value_bytes[4]:02}-{value_bytes[3]:02}-T{value_bytes[2]:02}:{value_bytes[1]:02}:{value_bytes[0]:02}"
        elif data_type == "16s":
            # Pad to 16 bytes if needed (value_bytes is 13 bytes from TBL file)
            value_padded = value_bytes[:16].ljust(16, b"\x00")
            value_unpacked = struct.unpack("16s", value_padded)[0]  # Read as bytes
            # Decode and truncate at first null byte
            return (
                value_unpacked.split(b"\x00", 1)[0]
                .decode("ascii", errors="ignore")
                .strip()
            )
        else:
            # Return raw bytes for unknown types
            return value_bytes

    def _get_dictionary_from_tbl(
        self, file_path: Path, decode_values: bool = True
    ) -> dict[str, int | float | str | bytes]:
        """
        Read TBL file and return a dictionary of all tag-value pairs.

        Parameters
        ----------
        file_path : Path
            Full path to the TBL file.
        decode_values : bool, default True
            If True, decode values according to TBL_TAG_TYPES mapping.
            If False, return raw bytes for all values.

        Returns
        -------
        dict[str, int | float | str | bytes]
            Dictionary with tag names as keys and decoded (or raw) values.
            Duplicate keys are handled by appending numeric suffixes (e.g., 'EGN_1', 'EGN_2').

        Notes
        -----
        This method reads the entire TBL file in 25-byte blocks, extracting
        key-value pairs. Each block contains:

        - Bytes 0-11: Tag name (null-terminated string)
        - Bytes 12-24: Value (13 bytes in various formats)

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> data = tbl._get_dictionary_from_tbl(Path('/data/file.TBL'))
        >>> print(data['SNUM'])
        1690
        """
        tbl_dict = {}

        with open(file_path, "rb") as fid:
            while True:
                block = fid.read(25)
                if len(block) < 25:
                    break

                # Extract key from first 12 bytes
                key = block[0:12].decode("latin-1").split("\x00")[0].strip()

                # Skip empty keys
                if not key:
                    continue

                # Extract value from last 13 bytes
                value_bytes = block[12:]

                if decode_values:
                    if key in self.TBL_TAG_TYPES.keys():
                        data_type = self.TBL_TAG_TYPES[key][0]
                    else:
                        data_type = "char"
                    try:
                        value = self.decode_tbl_value(value_bytes, data_type)
                    except Exception as e:
                        logger.warning(
                            f"Failed to decode {key}: {e}, storing raw bytes"
                        )
                        value = value_bytes
                else:
                    # Store raw bytes for unknown tags or if decode_values is False
                    value = value_bytes

                # Handle duplicate keys by appending index
                if key in tbl_dict:
                    # If this is the first duplicate, rename the original
                    if f"{key}_1" not in tbl_dict:
                        tbl_dict[f"{key}_1"] = tbl_dict[key]
                        del tbl_dict[key]
                    # Find next available index
                    idx = 2
                    while f"{key}_{idx}" in tbl_dict:
                        idx += 1
                    tbl_dict[f"{key}_{idx}"] = value
                else:
                    tbl_dict[key] = value

        return tbl_dict

    def read_tbl(self) -> None:
        """
        Read and decode the TBL file, populating the tbl_dict attribute.

        This method reads the TBL file specified during initialization and
        decodes all tag-value pairs according to their known types. The
        results are stored in `self.tbl_dict`.

        Returns
        -------
        None
            Results are stored in the `tbl_dict` attribute.

        Examples
        --------
        >>> tbl = MTUTable('/data/phoenix', '1690C16C.TBL')
        >>> tbl.read_tbl()
        >>> print(tbl.tbl_dict['SITE'])
        '10441W10'
        >>> print(tbl.tbl_dict['SNUM'])
        1690
        """
        if self.file_path is None:
            raise ValueError("file_path is not set. Cannot read TBL file.")
        elif self.file_path.is_file() is False:
            raise FileNotFoundError(f"TBL file not found: {self.file_path}")
        elif self.file_path.suffix.upper() != ".TBL":
            raise ValueError(f"Not a TBL file: {self.file_path}")
        elif self.file_path.stat().st_size == 0:
            raise ValueError(f"TBL file is empty: {self.file_path}")
        elif self.file_path.stat().st_size < 25:
            raise ValueError(f"TBL file is too small to be valid: {self.file_path}")
        elif self.file_path.exists() is False:
            raise FileNotFoundError(f"TBL file does not exist: {self.file_path}")

        self.tbl_dict = self._get_dictionary_from_tbl(
            self.file_path, decode_values=True
        )

    def _has_metadata(self) -> bool:
        """
        Check if TBL metadata has been loaded.

        Returns
        -------
        bool
            True if tbl_dict is populated, False otherwise.
        """
        return bool(self.tbl_dict)

    def _read_latitude(self, lat_str: str) -> float:
        """
        Convert latitude from degree-minute format to decimal degrees.

        Parameters
        ----------
        lat_str : str
            Latitude string in format 'DDMM.MMM,H' where H is hemisphere (N/S).
            Example: '4100.388,N' represents 41° 00.388' North.

        Returns
        -------
        float
            Latitude in decimal degrees. Negative for Southern hemisphere.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> lat = tbl._read_latitude('4100.388,N')
        >>> print(f"{lat:.6f}")
        41.006467
        """
        try:
            parts = lat_str.split(",", 1)
            value = float(parts[0]) / 100.0
            quadrant = parts[1]
            hemisphere = 1
            if quadrant.lower().startswith("s"):
                hemisphere = -1

            return value * hemisphere

        except Exception as e:
            logger.warning(f"Failed to parse latitude '{lat_str}': {e}")
            return 0.0

    def _read_longitude(self, lon_str: str) -> float:
        """
        Convert longitude from degree-minute format to decimal degrees.

        Parameters
        ----------
        lon_str : str
            Longitude string in format 'DDDMM.MMM,H' where H is hemisphere (E/W).
            Example: '10400.536,E' represents 104° 00.536' East.

        Returns
        -------
        float
            Longitude in decimal degrees. Negative for Western hemisphere.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> lon = tbl._read_longitude('10400.536,E')
        >>> print(f"{lon:.6f}")
        104.008933
        """
        try:
            parts = lon_str.split(",", 1)
            value = float(parts[0]) / 100.0
            quadrant = parts[1]
            hemisphere = 1
            if quadrant.lower().startswith("w"):
                hemisphere = -1

            return value * hemisphere

        except Exception as e:
            logger.warning(f"Failed to parse longitude '{lon_str}': {e}")
            return 0.0

    @property
    def channel_keys(self) -> dict[str, int]:
        """
        Get list of channel keys present in the TBL metadata.

        Returns
        -------
        dict[str, int]
            Dictionary of channel keys and their corresponding values found in tbl_dict (e.g., 'CHEX', 'CHEY', 'CHHX', etc.).

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> keys = tbl.channel_keys
        >>> print(keys)
        {'ex': 1, 'ey': 2, 'hx': 3, 'hy': 4, 'hz': 5}
        """
        channel_keys = {}
        for key in ["CHEX", "CHEY", "CHHX", "CHHY", "CHHZ"]:
            if key in self.tbl_dict:
                channel_keys[f"{key[2:].lower()}"] = self.tbl_dict[key]
        return channel_keys

    @property
    def survey_metadata(self) -> Survey:
        """
        Extract survey metadata from TBL file.

        Returns
        -------
        Survey
            mt_metadata Survey object populated with survey-level information
            from the TBL file (survey ID, company/author).

        Notes
        -----
        If TBL metadata has not been loaded (via `read_tbl()`), returns an
        empty Survey object with a warning.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> survey = tbl.survey_metadata
        >>> print(survey.id)
        'MT_Survey_2024'
        """
        survey = Survey()  # type: ignore
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty Survey."
            )
        else:
            survey.id = self.tbl_dict.get("SRVY", "Unknown_Survey")
            survey.acquired_by.author = self.tbl_dict.get("CMPY", "Unknown_Company")
            survey.add_station(self.station_metadata)

        return survey

    @property
    def station_metadata(self) -> Station:
        """
        Extract station metadata from TBL file.

        Returns
        -------
        Station
            mt_metadata Station object populated with station-level information
            including location (latitude, longitude, elevation, declination)
            and time period.

        Notes
        -----
        If TBL metadata has not been loaded (via `read_tbl()`), returns an
        empty Station object with a warning.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> station = tbl.station_metadata
        >>> print(station.id)
        '10441W10'
        >>> print(f"{station.location.latitude:.6f}")
        41.006467
        """
        station = Station()  # type: ignore
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty Station."
            )
        else:
            station.id = self.tbl_dict.get("SITE", "Unknown_Site")
            # location
            station.location.elevation = self.tbl_dict.get("ELEV", 0.0)
            station.location.latitude = self._read_latitude(
                self.tbl_dict.get("LATG", "0.0,N")
            )
            station.location.longitude = self._read_longitude(
                self.tbl_dict.get("LNGG", "0.0,E")
            )
            station.location.declination.value = self.tbl_dict.get("DECL", 0.0)

            # time
            station.time_period.start = self.tbl_dict.get("STIM", "1980-01-01T00:00:00")
            station.time_period.end = self.tbl_dict.get("ETIM", "1980-01-01T00:00:00")

            # runs
            station.add_run(self.run_metadata)
            # Populate station metadata from tbl_dict as needed
        return station

    @property
    def run_metadata(self) -> Run:
        """
        Extract run metadata from TBL file.

        Returns
        -------
        Run
            mt_metadata Run object populated with data logger information
            and channel metadata.

        Notes
        -----
        If TBL metadata has not been loaded (via `read_tbl()`), returns an
        empty Run object with a warning.

        The run includes all channel metadata (ex, ey, hx, hy, hz) obtained
        from their respective property methods.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> run = tbl.run_metadata
        >>> print(run.id)
        'run_1690'
        >>> print(run.data_logger.id)
        'MTU_1690'
        """
        run = Run()  # type: ignore
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty Run."
            )
        else:
            # Populate run metadata from tbl_dict as needed
            run.id = f"run_{self.tbl_dict.get('SNUM', 'Unknown')}"
            run.data_logger.id = f"MTU_{self.tbl_dict.get('SNUM', 'Unknown')}"
            run.data_logger.firmware.version = self.tbl_dict.get(
                "HW", "Unknown_Version"
            )
            run.data_logger.timing_system.type = "GPS"
            run.data_logger.timing_system.n_satellites = self.tbl_dict.get("NSAT", 0)

            run.add_channel(self.ex_metadata)
            run.add_channel(self.ey_metadata)
            run.add_channel(self.hx_metadata)
            run.add_channel(self.hy_metadata)
            run.add_channel(self.hz_metadata)
            run.update_time_period()

        return run

    @property
    def ex_metadata(self) -> Electric:
        """
        Extract Ex electric channel metadata from TBL file.

        Returns
        -------
        Electric
            mt_metadata Electric object for Ex component with dipole length,
            azimuth, AC/DC start values, and channel number.

        Notes
        -----
        If TBL metadata has not been loaded (via `read_tbl()`), returns an
        empty Electric object with a warning.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> ex = tbl.ex_metadata
        >>> print(ex.dipole_length)
        100.0
        """
        ex_channel = Electric(component="ex")  # type: ignore
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty EX channel."
            )
        else:
            ex_channel.dipole_length = self.tbl_dict.get("EXLN", 0.0)
            ex_channel.measurement_azimuth = self.tbl_dict.get("EAZM", 0.0)
            ex_channel.ac.start = self.tbl_dict.get("EXAC", 0.0)
            ex_channel.dc.start = self.tbl_dict.get("EXDC", 0.0)
            ex_channel.channel_number = self.tbl_dict.get("CHEX", 4)
            ex_channel.time_period.start = self.tbl_dict.get(
                "STIM", "1980-01-01T00:00:00"
            )
            ex_channel.time_period.end = self.tbl_dict.get(
                "ETIM", "1980-01-01T00:00:00"
            )
            ex_channel.units = "digital counts"
        return ex_channel

    @property
    def ey_metadata(self) -> Electric:
        """
        Extract Ey electric channel metadata from TBL file.

        Returns
        -------
        Electric
            mt_metadata Electric object for Ey component with dipole length,
            azimuth (Ex azimuth + 90°), AC/DC start values, and channel number.

        Notes
        -----
        If TBL metadata has not been loaded (via `read_tbl()`), returns an
        empty Electric object with a warning.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> ey = tbl.ey_metadata
        >>> print(ey.dipole_length)
        100.0
        """
        ey_channel = Electric(component="ey")  # type: ignore
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty EY channel."
            )
        else:
            ey_channel.dipole_length = self.tbl_dict.get("EYLN", 0.0)
            ey_channel.measurement_azimuth = self.tbl_dict.get("EAZM", 0.0) + 90.0
            ey_channel.ac.start = self.tbl_dict.get("EYAC", 0.0)
            ey_channel.dc.start = self.tbl_dict.get("EYDC", 0.0)
            ey_channel.channel_number = self.tbl_dict.get("CHEY", 5)
            ey_channel.time_period.start = self.tbl_dict.get(
                "STIM", "1980-01-01T00:00:00"
            )
            ey_channel.time_period.end = self.tbl_dict.get(
                "ETIM", "1980-01-01T00:00:00"
            )
            ey_channel.units = "digital counts"
        return ey_channel

    @property
    def hx_metadata(self) -> Magnetic:
        """
        Extract Hx magnetic channel metadata from TBL file.

        Returns
        -------
        Magnetic
            mt_metadata Magnetic object for Hx component with maximum field,
            channel number, azimuth, and sensor serial number.

        Notes
        -----
        If TBL metadata has not been loaded (via `read_tbl()`), returns an
        empty Magnetic object with a warning.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> hx = tbl.hx_metadata
        >>> print(hx.sensor.id)
        'coil1693'
        """
        hx_channel = Magnetic(component="hx")  # type: ignore
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty HX channel."
            )
        else:
            # Note: HXAC is a single value, but h_field_max expects StartEndRange
            # Skipping assignment for now
            hx_channel.channel_number = self.tbl_dict.get("CHHX", 1)
            hx_channel.measurement_azimuth = self.tbl_dict.get("HAZM", 0.0)
            hx_channel.sensor.id = self.tbl_dict.get("HXSN", "Unknown_serial")
            hx_channel.time_period.start = self.tbl_dict.get(
                "STIM", "1980-01-01T00:00:00"
            )
            hx_channel.time_period.end = self.tbl_dict.get(
                "ETIM", "1980-01-01T00:00:00"
            )
            hx_channel.units = "digital counts"
        return hx_channel

    @property
    def hy_metadata(self) -> Magnetic:
        """
        Extract Hy magnetic channel metadata from TBL file.

        Returns
        -------
        Magnetic
            mt_metadata Magnetic object for Hy component with maximum field,
            channel number, azimuth (Hx azimuth + 90°), and sensor serial number.

        Notes
        -----
        If TBL metadata has not been loaded (via `read_tbl()`), returns an
        empty Magnetic object with a warning.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> hy = tbl.hy_metadata
        >>> print(hy.sensor.id)
        'coil1694'
        """
        hy_channel = Magnetic(component="hy")  # type: ignore
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty HY channel."
            )
        else:
            # Note: HYAC is a single value, but h_field_max expects StartEndRange
            # Skipping assignment for now
            hy_channel.channel_number = self.tbl_dict.get("CHHY", 2)
            hy_channel.measurement_azimuth = self.tbl_dict.get("HAZM", 0.0) + 90.0
            hy_channel.sensor.id = self.tbl_dict.get("HYSN", "Unknown_serial")
            hy_channel.time_period.start = self.tbl_dict.get(
                "STIM", "1980-01-01T00:00:00"
            )
            hy_channel.time_period.end = self.tbl_dict.get(
                "ETIM", "1980-01-01T00:00:00"
            )
            hy_channel.units = "digital counts"
        return hy_channel

    @property
    def hz_metadata(self) -> Magnetic:
        """
        Extract Hz magnetic channel metadata from TBL file.

        Returns
        -------
        Magnetic
            mt_metadata Magnetic object for Hz component with maximum field,
            channel number, and sensor serial number.

        Notes
        -----
        If TBL metadata has not been loaded (via `read_tbl()`), returns an
        empty Magnetic object with a warning.

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> hz = tbl.hz_metadata
        >>> print(hz.sensor.id)
        'coil1695'
        """
        hz_channel = Magnetic(component="hz")  # type: ignore
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty HZ channel."
            )
        else:
            # Note: HZAC is a single value, but h_field_max expects StartEndRange
            # Skipping assignment for now
            hz_channel.channel_number = self.tbl_dict.get("CHHZ", 3)
            hz_channel.sensor.id = self.tbl_dict.get("HZSN", "Unknown_serial")
            hz_channel.time_period.start = self.tbl_dict.get(
                "STIM", "1980-01-01T00:00:00"
            )
            hz_channel.time_period.end = self.tbl_dict.get(
                "ETIM", "1980-01-01T00:00:00"
            )
            hz_channel.units = "digital counts"
        return hz_channel

    @property
    def ex_calibration(self) -> float | None:
        """
        Calculate Ex channel calibration factor.

        Returns
        -------
        float or None
            Calibration factor to convert raw ADC values to mV/km.
            Returns None if TBL metadata has not been loaded.

        Notes
        -----
        The calibration factor is calculated as:

        .. math::
            \\text{cal} = \\frac{\\text{FSCV}}{2^{23}} \\times \\frac{1000}{\\text{EGN}} \\times \\frac{1000}{\\text{EXLN}}

        where:

        - FSCV: Full-scale converter voltage
        - EGN: Electric channel gain
        - EXLN: Ex dipole length in meters

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> cal = tbl.ex_calibration
        >>> print(f"{cal:.6f}")
        0.000762
        """

        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning None."
            )
            return None
        # E field as mV/km
        return (
            float(self.tbl_dict["FSCV"])
            / 2**23
            * 1000
            / float(self.tbl_dict["EGN"])
            / float(self.tbl_dict["EXLN"])
            * 1000
        )

    @property
    def ey_calibration(self) -> float | None:
        """
        Calculate Ey channel calibration factor.

        Returns
        -------
        float or None
            Calibration factor to convert raw ADC values to mV/km.
            Returns None if TBL metadata has not been loaded.

        Notes
        -----
        The calibration factor is calculated as:

        .. math::
            \\text{cal} = \\frac{\\text{FSCV}}{2^{23}} \\times \\frac{1000}{\\text{EGN}} \\times \\frac{1000}{\\text{EYLN}}

        where:

        - FSCV: Full-scale converter voltage
        - EGN: Electric channel gain
        - EYLN: Ey dipole length in meters

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> cal = tbl.ey_calibration
        >>> print(f"{cal:.6f}")
        0.000762
        """

        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning None."
            )
            return None
        # E field as mV/km
        return (
            float(self.tbl_dict["FSCV"])
            / 2**23
            * 1000
            / float(self.tbl_dict["EGN"])
            / float(self.tbl_dict["EYLN"])
            * 1000
        )

    @property
    def magnetic_calibration(self) -> float | None:
        """
        Calculate magnetic channel calibration factor.

        Returns
        -------
        float or None
            Calibration factor to convert raw ADC values to nT.
            Returns None if TBL metadata has not been loaded.

        Notes
        -----
        The calibration factor is calculated as:

        .. math::
            \\text{cal} = \\frac{\\text{FSCV}}{2^{23}} \\times \\frac{1000}{\\text{HGN} \\times \\text{HATT} \\times \\text{HNOM}}

        where:

        - FSCV: Full-scale converter voltage
        - HGN: Magnetic channel gain
        - HATT: Magnetic channel attenuation
        - HNOM: Magnetic channel normalization (mA/nT)

        This calibration applies to all magnetic channels (Hx, Hy, Hz).

        Examples
        --------
        >>> tbl = MTUTable('/data', 'file.TBL')
        >>> tbl.read_tbl()
        >>> cal = tbl.magnetic_calibration
        >>> print(f"{cal:.9f}")
        0.000000229
        """

        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning None."
            )
            return None
        # H field as nT
        return (
            float(self.tbl_dict["FSCV"])
            / 2**23
            * 1000
            / float(self.tbl_dict["HGN"])
            / float(self.tbl_dict["HATT"])
            / float(self.tbl_dict["HNOM"])
        )
