Welcome to MTH5's documentation!
======================================
.. image:: source/images/mth5_logo.png
	:align: center

|

MTH5 is an HDF5 data container for magnetotelluric time series data, but could be extended to other data types.  This package provides tools for reading/writing/manipulating MTH5 files.

MTH5 uses `h5py <https://www.h5py.org/>`__  to interact with the HDF5 file, `xarray <http://xarray.pydata.org/en/stable/>`__ to interact with the data in a nice way, and all metadata use `mt_metadata <https://github.com/kujaku11/mt_metadata>`__. 

This project is in cooperation with the Incorporated Research Institutes of Seismology, the U.S. Geological Survey, and other collaborators.  Facilities of the IRIS Consortium are supported by the National Science Foundation’s Seismological Facilities for the Advancement of Geoscience (SAGE) Award under Cooperative Support Agreement EAR-1851048.  USGS is partially funded through the Community for Data Integration and IMAGe through the Minerals Resources Program. 

|

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    readme
    installation
    contributing
    authors
    history
    source/gotchas
    usage
    usage_v2
    source/structure
    source/ts
    source/file_readers
    source/conventions
    source/examples
	
.. toctree::
	:maxdepth: 1
	:caption: Metadata Standards
	
	source/mt_metadata_guide
	

.. toctree::
    :maxdepth: 1
    :caption: Packages

    source/mth5

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
