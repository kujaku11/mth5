==========
GOTCHAS
==========

There are some gotchas or things you should understand when using HDF5 files as well as MTH5

Compression
""""""""""""

Compression can slow down making a MTH5 file, so you should understand the compression parameters.  See `H5 Compression <https://pythonhosted.org/hdf5storage/compression.html>`__ and `DataSets <https://docs.h5py.org/en/stable/high/dataset.html>`__ for more information.

Compression is set in MTH5 when you instatiate an MTH5 object

	>>> m = MTH5(shuffle=None, fletcher32=None, compression=None, compression_opts=None)
	
The compression parameters will be validated using `mth5.helpers.validate_compression`

Datasets can use chunks, which by default is set to True, which lets h5py pick the most efficient way to chunk the data.  

Lossless compression filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GZIP filter (``"gzip"``)
    Available with every installation of HDF5, so it's best where portability is
    required.  Good compression, moderate speed.  ``compression_opts`` sets the
    compression level and may be an integer from 0 to 9, default is 4.


LZF filter (``"lzf"``)
    Available with every installation of h5py (C source code also available).
    Low to moderate compression, very fast.  No options.


SZIP filter (``"szip"``)
    Patent-encumbered filter used in the NASA community.  Not available with all
    installations of HDF5 due to legal reasons.  Consult the HDF5 docs for filter
    options.

Logging
"""""""""""

Logging is great, but can have dramatic effects on performance, mainly because I'm new to logging and probably haven't written them most efficiently.  By default the logging level is set to INFO.  This seems to run as you might expect with slight overhead.  If you change the logging level to DEBUG expect a slow down.  You should only do this if you are a developer or are curious as to why something looks weird.

