import struct
from pathlib import Path

from loguru import logger
from mt_metadata.timeseries import Survey


def find_tbl_tag(fid, tag, inum):
    """
    find_tbl_tag - find the position of a certain tag at its "inum"th occurrence
    in file - and returns the tag values, for the legacy Phoenix MTU-5A TBL
    format

    =======================================================================
    note:
    yes, it's a bit silly to search for the tag each time from the beginning
    it doesn't matter too much as the file is quite small
    Hao
    2012.07.04
    Beijing
    =======================================================================
    """
    # firstly return to the beginning of the file
    fid.seek(0, 0)
    # read a first tag group
    ctemp = fid.read(25)
    n = 0
    # continue reading, until we got the "inum" occurrence
    while tag.encode() not in ctemp or n != inum:
        # the size (tag + value) is always 25 bytes
        ctemp = fid.read(25)
        if tag.encode() in ctemp:
            n += 1
        if len(ctemp) == 0:
            logger.warning(f"Tag {tag} not found in TBL file.")
            return None, 0
    pos = fid.tell()
    return ctemp, pos


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
            "TEMP": ("double", "Temperature"),
            "TERR": ("double", "Temperature error"),
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
            "CHEX": ("char", "EX channel type"),
            "CHEY": ("char", "EY channel type"),
            "CHHX": ("char", "HX channel type"),
            "CHHY": ("char", "HY channel type"),
            "CHHZ": ("char", "HZ channel type"),
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
            return f"{value_bytes[4]:02}-{value_bytes[3]:02}-20{value_bytes[5]:02}T{value_bytes[2]:02}:{value_bytes[1]:02}:{value_bytes[0]:02}"
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

    @property
    def survey_metadata(self):
        survey = Survey()
        # Populate survey metadata from tbl_dict as needed
        return survey


def read_tbl(fpath, fname):
    """
    read_tbl - reads a (binary) TBL table file of the legacy Phoenix format
    (MTU-5A) and output the "info" metadata dictionary

    Parameters:
        fpath: path to the tbl
        fname: name of the tbl file (including extensions)

    Returns:
        info: output dict of the TBL metadata
    """
    info = {}

    # first open the file
    filepath = Path(fpath) / fname
    with open(filepath, "rb") as fid:
        # ========================= site basic info  ========================== #
        find_tbl_tag(fid, "SNUM", 1)
        fid.seek(-13, 1)
        info["SNUM"] = struct.unpack("<i", fid.read(4))[0]

        ctemp, _ = find_tbl_tag(fid, "SITE", 1)
        info["SITE"] = ctemp[12:24].decode("latin-1").strip("\x00")

        ctemp, _ = find_tbl_tag(fid, "FILE", 1)
        info["FILE"] = ctemp[12:24].decode("latin-1").strip("\x00")

        ctemp, _ = find_tbl_tag(fid, "CMPY", 1)
        info["CMPY"] = ctemp[12:24].decode("latin-1").strip("\x00")

        ctemp, _ = find_tbl_tag(fid, "SRVY", 1)
        info["SRVY"] = ctemp[12:24].decode("latin-1").strip("\x00")

        ctemp, _ = find_tbl_tag(fid, "LATG", 1)
        info["LATG"] = ctemp[12:24].decode("latin-1").strip("\x00")

        ctemp, _ = find_tbl_tag(fid, "LNGG", 1)
        info["LONG"] = ctemp[12:24].decode("latin-1").strip("\x00")

        find_tbl_tag(fid, "ELEV", 1)
        fid.seek(-13, 1)
        info["ELEV"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "NREF", 1)
        fid.seek(-13, 1)
        info["NREF"] = struct.unpack("<i", fid.read(4))[0]

        # ==================== starting and ending time  ====================== #
        # output as a string
        ctemp, _ = find_tbl_tag(fid, "STIM", 1)
        info[
            "STIM"
        ] = f"{ctemp[16]}-{ctemp[15]}-20{ctemp[17]} {ctemp[14]}:{ctemp[13]}:{ctemp[12]}"

        # output as a string
        ctemp, _ = find_tbl_tag(fid, "ETIM", 1)
        info[
            "ETIM"
        ] = f"{ctemp[16]}-{ctemp[15]}-20{ctemp[17]} {ctemp[14]}:{ctemp[13]}:{ctemp[12]}"

        # ======================== E and H channels  ========================== #
        find_tbl_tag(fid, "EXLN", 1)
        fid.seek(-13, 1)
        info["EXLN"] = struct.unpack("<d", fid.read(8))[0]

        find_tbl_tag(fid, "EYLN", 1)
        fid.seek(-13, 1)
        info["EYLN"] = struct.unpack("<d", fid.read(8))[0]

        # find_tbl_tag(fid,'EZLN',1);
        # fid.seek(-13, 1)
        # info['EZLN'] = struct.unpack('<d', fid.read(8))[0]

        ctemp, _ = find_tbl_tag(fid, "HXSN", 1)
        info["HXSN"] = ctemp[12:24].decode("latin-1").strip("\x00")

        ctemp, _ = find_tbl_tag(fid, "HYSN", 1)
        info["HYSN"] = ctemp[12:24].decode("latin-1").strip("\x00")

        ctemp, _ = find_tbl_tag(fid, "HZSN", 1)
        info["HZSN"] = ctemp[12:24].decode("latin-1").strip("\x00")

        find_tbl_tag(fid, "EAZM", 1)
        fid.seek(-13, 1)
        info["EAZM"] = struct.unpack("<d", fid.read(8))[0]

        find_tbl_tag(fid, "HAZM", 1)
        fid.seek(-13, 1)
        info["HAZM"] = struct.unpack("<d", fid.read(8))[0]

        # ================== L3, L4 and L5 sample parameter =================== #
        find_tbl_tag(fid, "HSMP", 1)
        fid.seek(-13, 1)
        info["HSMP"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "L3NS", 1)
        fid.seek(-13, 1)
        info["L3NS"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "L4NS", 1)
        fid.seek(-13, 1)
        info["L4NS"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "SRL3", 1)
        fid.seek(-13, 1)
        info["SRL3"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "SRL4", 1)
        fid.seek(-13, 1)
        info["SRL4"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "SRL5", 1)
        fid.seek(-13, 1)
        info["SRL5"] = struct.unpack("<i", fid.read(4))[0]

        # ======================== gain and filtering ========================= #
        ctemp, _ = find_tbl_tag(fid, "LFRQ", 1)
        info["LFRQ"] = ctemp[12]

        find_tbl_tag(fid, "EGNC", 1)
        fid.seek(-13, 1)
        info["EGNC"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "HGNC", 1)
        fid.seek(-13, 1)
        info["HGNC"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "EGN", 2)
        fid.seek(-13, 1)
        info["EGN"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "HGN", 2)
        fid.seek(-13, 1)
        info["HGN"] = struct.unpack("<i", fid.read(4))[0]

        find_tbl_tag(fid, "HATT", 1)
        fid.seek(-13, 1)
        info["HATT"] = struct.unpack("<d", fid.read(8))[0]

        find_tbl_tag(fid, "HNOM", 1)
        fid.seek(-13, 1)
        info["HNOM"] = struct.unpack("<d", fid.read(8))[0]

        find_tbl_tag(fid, "TCMB", 1)
        fid.seek(-13, 1)
        info["TCMB"] = struct.unpack("<B", fid.read(1))[0]

        find_tbl_tag(fid, "TALS", 1)
        fid.seek(-13, 1)
        info["TALS"] = struct.unpack("<B", fid.read(1))[0]

        find_tbl_tag(fid, "LPFR", 1)
        fid.seek(-13, 1)
        info["LPFR"] = struct.unpack("<B", fid.read(1))[0]

        find_tbl_tag(fid, "ACDC", 1)
        fid.seek(-13, 1)
        info["ACDC"] = struct.unpack("<B", fid.read(1))[0]

        find_tbl_tag(fid, "FSCV", 1)
        fid.seek(-13, 1)
        info["FSCV"] = struct.unpack("<d", fid.read(8))[0]

    # ======================================================================= #
    # now the file is closed automatically by the context manager
    return info


if __name__ == "__main__":
    # simple test
    import sys

    if len(sys.argv) >= 3:
        info = read_tbl(sys.argv[1], sys.argv[2])
        for key, value in info.items():
            print(f"{key}: {value}")
