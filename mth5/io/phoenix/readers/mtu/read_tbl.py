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
    HSMP: L3 and L4 time slot in second （MTU-5A) or minute (MTU-5P),
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

    Hao
    2012.07.04
    Beijing
    =======================================================================
    """

    def __init__(self, fpath, fname):
        self.fpath = Path(fpath)
        self.fname = fname
        self.file = self.fpath / self.fname
        self.tbl_dict = {}

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
            # Additional tags that may appear in TBL files
            "DECL": ("double", "Declination"),
            "TSTV": ("double", "Test voltage"),
            "VER": ("char", "Version"),
            "HW": ("char", "Hardware"),
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

    def decode_tbl_value(self, value_bytes, data_type):
        """
        Decode TBL value bytes based on the specified data type.

        Parameters:
            value_bytes: bytes (13 bytes from position 12-24 in the 25-byte block)
            data_type: str - type of the data ('int', 'double', 'char', 'byte', 'time')

        Returns:
            Decoded value in appropriate Python type
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
        else:
            # Return raw bytes for unknown types
            return value_bytes

    def _get_dictionary_from_tbl(self, file_path, decode_values=True):
        """
        Read TBL file and return a dictionary of all tag-value pairs.

        Parameters:
            file_path: Path to the TBL file
            decode_values: bool - If True, decode values according to TBL_TAG_TYPES.
                        If False, return raw bytes.

        Returns:
            dict: Dictionary with tag names as keys and decoded (or raw) values
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

    def read_tbl(self):
        self.tbl_dict = self._get_dictionary_from_tbl(self.file, decode_values=True)

    def _has_metadata(self):
        return bool(self.tbl_dict)

    def _read_latitude(self, lat_str):
        """Convert degree-minute string to decimal degrees."""
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

    def _read_longitude(self, lon_str):
        """Convert degree-minute string to decimal degrees."""
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
    def survey_metadata(self):
        survey = Survey()
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
    def station_metadata(self):
        station = Station()
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
                self.tbl_dict.get("LONG", "0.0,E")
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
    def run_metadata(self):
        run = Run()
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

        return run

    @property
    def ex_metadata(self):
        ex_channel = Electric(component="ex")
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
        return ex_channel

    @property
    def ey_metadata(self):
        ey_channel = Electric(component="ey")
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
        return ey_channel

    @property
    def hx_metadata(self):
        hx_channel = Magnetic(component="hx")
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty HX channel."
            )
        else:
            hx_channel.h_field_max = self.tbl_dict.get("HXAC", 0.0)
            hx_channel.channel_number = self.tbl_dict.get("CHHX", 1)
            hx_channel.measurement_azimuth = self.tbl_dict.get("HAZM", 0.0)
            hx_channel.sensor.id = self.tbl_dict.get("HXSN", "Unknown_serial")
        return hx_channel

    @property
    def hy_metadata(self):
        hy_channel = Magnetic(component="hy")
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty HY channel."
            )
        else:
            hy_channel.h_field_max = self.tbl_dict.get("HYAC", 0.0)
            hy_channel.channel_number = self.tbl_dict.get("CHHY", 2)
            hy_channel.measurement_azimuth = self.tbl_dict.get("HAZM", 0.0) + 90.0
            hy_channel.sensor.id = self.tbl_dict.get("HYSN", "Unknown_serial")
        return hy_channel

    @property
    def hz_metadata(self):
        hz_channel = Magnetic(component="hz")
        if not self._has_metadata():
            logger.warning(
                "No TBL metadata loaded. Call read_tbl() first. Returning empty HZ channel."
            )
        else:
            hz_channel.h_field_max = self.tbl_dict.get("HZAC", 0.0)
            hz_channel.channel_number = self.tbl_dict.get("CHHZ", 3)
            hz_channel.sensor.id = self.tbl_dict.get("HZSN", "Unknown_serial")
        return hz_channel
