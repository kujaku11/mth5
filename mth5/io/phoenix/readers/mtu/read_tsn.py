"""
=======================================================================
original comments from MATLAB script:

read_tsn - reads a (binary) TS file of the legacy Phoenix MTU-5A instrument
(TS2, TS3, TS4, TS5) and the even older V5-2000 system (TSL, TSH), and
output the "ts" array and "tag" metadata dictionary.

=======================================================================
Parameters:
    fpath: path to the TS file
    fname: name of the TS file (including extensions)

Returns:
    ts:    output array of the TS data
    tag:   output dict of the TSn metadata

=======================================================================
definition of the TS tag (or what I guessed after reading the user manual
and fiddling with their files)
0-7   UTC time of first scan in the record.
8-9   instrument serial number (16-bit integer)
10-11 number of scans in the record (16-bit integer)
12    number of channels per scan
13    tag length (TSn) or tag length code (TSH, TSL)
14    status code
15    bit-wise saturation flags (please note that the older TSH/L tag
      ends here )
16    reserved for future indication of different tag and/or sample
      formats
17    sample length in bytes
18-19 sample rate (in units defined by byte 20)
20    units of sample rate
21    clock status
22-25 clock error in seconds
26-32 reserved; must be 0

=======================================================================
notes on the TS format of TSn files:
The binary TS file consists of several data blocks, each contains a data
tag and a number of records in it.
Each time record consists of three bytes (24 bit), let's name them byte1,
byte2, and byte3:
the ts record (int) should be (+/-) (byte3*65536 + byte2*256 + byte1)

Hao
2012.07.04
Beijing
=======================================================================
"""

import os

import numpy as np


def getsign24(x):
    """
    a simple function to calculate the sign for a 24 bit number
    I should have made it in-line
    """
    x = np.array(x, dtype=np.int32)
    x[x > 2**23 - 1] = x[x > 2**23 - 1] - 2**24
    return x


def read_tsn(fpath, fname):
    """
    read_tsn - reads a (binary) TS file of the legacy Phoenix MTU-5A instrument
    (TS2, TS3, TS4, TS5) and the even older V5-2000 system (TSL, TSH), and
    output the "ts" array and "tag" metadata dictionary.

    Parameters:
        fpath: path to the TS file
        fname: name of the TS file (including extensions)

    Returns:
        ts:    output numpy array of the TS data
        tag:   output dict of the TSn metadata
    """
    # try opening the ts data file
    filepath = os.path.join(fpath, fname)
    print(f"# opening file: {fname}")

    with open(filepath, "rb") as TSfid:
        # some constants used for format conversion
        p16 = 2**16
        p8 = 2**8

        # scan through time series
        # firstly reading a 32 Byte header info
        s_byte = TSfid.read(1)  # Starting second
        if len(s_byte) == 0:
            return None, None
        s = s_byte[0]
        m = TSfid.read(1)[0]  # Starting minute
        h = TSfid.read(1)[0]  # Starting hour
        d = TSfid.read(1)[0]  # Starting day
        l = TSfid.read(1)[0]  # Starting month
        y = TSfid.read(1)[0]  # Starting year
        TSfid.read(1)  # skip the Starting weekday
        c = TSfid.read(1)[0]  # Starting century(-1)

        # box series number (16-bit integer)
        bnum = int.from_bytes(TSfid.read(2), byteorder="little", signed=False)
        # Number of scans in a data block (16-bit integer)
        Nscan = int.from_bytes(TSfid.read(2), byteorder="little", signed=False)
        # Number of channels in a record
        Nch = TSfid.read(1)[0]
        # length of the tag
        Taglen = TSfid.read(1)[0]

        if Taglen != 32:
            tstype = "V5-2000"
        else:
            tstype = "MTU-5"
        print(f"# TS type is: {tstype}")

        Fs = 0  # Initialize sampling frequency
        if Taglen == 32:
            TSfid.seek(4, 1)  # skip to sampling frequency
            # Sampling frequency (16-bit integer, little endian)
            Fs = int.from_bytes(TSfid.read(2), byteorder="little", signed=False)
            TSfid.seek(12, 1)  # skip some (unknown) head info...
            print(f"# sampling frequency is {Fs} Hz")
            print(f"# number of records is {Nscan} in each data block")

        # now go to the end of the file...
        TSfid.seek(0, 2)
        file_size = TSfid.tell()
        # number of data blocks in the file
        Nblock = round(file_size / (Nscan * Nch * 3 + 32))

        # preallocate some memory for ts
        ts = np.zeros((Nch, Nscan * Nblock), dtype=np.float64)
        print(f"# total {Nblock} block(s) found in current file")

        # now go back to the beginning of the file...
        TSfid.seek(0, 0)

        for iblock in range(Nblock):
            # now start loading the data
            # here=TSfid.tell()  # for debug
            TSfid.seek(32, 1)  # skip the file tag
            # 3*5 = 15 Byte per record
            data = np.frombuffer(TSfid.read(Nch * 3 * Nscan), dtype=np.uint8)

            if len(data) == 0:
                print("# warning: no data read in current block...")
                break

            # Reshape data to [Nch*3, Nscan] (Fortran order to match MATLAB)
            # Convert to int32 to avoid overflow during multiplication
            data = data.reshape((Nscan, Nch * 3)).T.astype(np.int32)

            start_idx = iblock * Nscan
            end_idx = (iblock + 1) * Nscan

            if Nch == 3:
                ts[0, start_idx:end_idx] = getsign24(
                    data[2, :] * p16 + data[1, :] * p8 + data[0, :]
                )  # Ex1
                ts[1, start_idx:end_idx] = getsign24(
                    data[5, :] * p16 + data[4, :] * p8 + data[3, :]
                )  # Ex2
                ts[2, start_idx:end_idx] = getsign24(
                    data[8, :] * p16 + data[7, :] * p8 + data[6, :]
                )  # Ex3
            elif Nch == 4:
                ts[0, start_idx:end_idx] = getsign24(
                    data[2, :] * p16 + data[1, :] * p8 + data[0, :]
                )  # Ex1
                ts[1, start_idx:end_idx] = getsign24(
                    data[5, :] * p16 + data[4, :] * p8 + data[3, :]
                )  # Ex2
                ts[2, start_idx:end_idx] = getsign24(
                    data[8, :] * p16 + data[7, :] * p8 + data[6, :]
                )  # Ex3
                ts[3, start_idx:end_idx] = getsign24(
                    data[14, :] * p16 + data[13, :] * p8 + data[12, :]
                )  # Hy
            elif Nch == 5:
                ts[0, start_idx:end_idx] = getsign24(
                    data[2, :] * p16 + data[1, :] * p8 + data[0, :]
                )  # Ex
                ts[1, start_idx:end_idx] = getsign24(
                    data[5, :] * p16 + data[4, :] * p8 + data[3, :]
                )  # Ey
                ts[2, start_idx:end_idx] = getsign24(
                    data[8, :] * p16 + data[7, :] * p8 + data[6, :]
                )  # Hx
                ts[3, start_idx:end_idx] = getsign24(
                    data[11, :] * p16 + data[10, :] * p8 + data[9, :]
                )  # Hy
                ts[4, start_idx:end_idx] = getsign24(
                    data[14, :] * p16 + data[13, :] * p8 + data[12, :]
                )  # Hz
            elif Nch == 6:
                ts[0, start_idx:end_idx] = getsign24(
                    data[2, :] * p16 + data[1, :] * p8 + data[0, :]
                )  # Ex
                ts[1, start_idx:end_idx] = getsign24(
                    data[5, :] * p16 + data[4, :] * p8 + data[3, :]
                )  # Ey
                ts[2, start_idx:end_idx] = getsign24(
                    data[8, :] * p16 + data[7, :] * p8 + data[6, :]
                )  # Ez
                ts[3, start_idx:end_idx] = getsign24(
                    data[11, :] * p16 + data[10, :] * p8 + data[9, :]
                )  # Hx
                ts[4, start_idx:end_idx] = getsign24(
                    data[14, :] * p16 + data[13, :] * p8 + data[12, :]
                )  # Hy
                ts[5, start_idx:end_idx] = getsign24(
                    data[17, :] * p16 + data[16, :] * p8 + data[15, :]
                )  # Hz

    print("# finish reading time series...")

    # Build the tag dictionary
    tag = {
        "boxnum": bnum,
        "tstype": tstype,
        "Fs": Fs,
        "Nch": Nch,
        "Nscan": Nscan,
        "Tstr": [(c * 100) + y, l, d, h, m, s],
        "Tlen": Nscan / Fs if Fs > 0 else 0,
        "Nblock": Nblock,
    }

    return ts, tag


if __name__ == "__main__":
    # simple test
    import sys

    if len(sys.argv) >= 3:
        ts, tag = read_tsn(sys.argv[1], sys.argv[2])
        if tag is not None:
            print("\nTag information:")
            for key, value in tag.items():
                print(f"{key}: {value}")
            print(f"\nTime series shape: {ts.shape}")
