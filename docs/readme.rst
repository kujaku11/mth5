.. image:: images/mth5_logo.png
   :alt: MTH5 Logo
   :align: center


The goal of **MTH5** is to develop a standard format and tools for archiving 
magnetotelluric (MT) time series data.

The preferred format is HDF5 and has been adopted to conform to MT data,
something that has been needed in the EM community for some time. The
module mth5 contains reading/writing capabilities and will contain tools
for retrieving data in useful ways to work with processing codes.

The metadata follows the standards proposed by the `IRIS-PASSCAL MT
Software working
group <https://www.iris.edu/hq/about_iris/governance/mt_soft>`__ and
documented in `MT Metadata
Standards <https://github.com/kujaku11/MTarchive/blob/tables/docs/mt_metadata_guide.pdf>`__.

.. note:: This is a work in progress. Feel free to comment or send me a message at jpeacock@usgs.gov on the data format.

MTH5 Format
-----------

-  The basic format of MTH5 is illustrated below, where metadata is
   attached at each level.

.. figure:: images/example_mt_file_structure.png
   :alt: MTH5 Format
   :align: center
