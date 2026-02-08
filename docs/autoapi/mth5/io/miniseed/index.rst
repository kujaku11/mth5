mth5.io.miniseed
================

.. py:module:: mth5.io.miniseed


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/mth5/io/miniseed/miniseed/index


Functions
---------

.. autoapisummary::

   mth5.io.miniseed.read_miniseed


Package Contents
----------------

.. py:function:: read_miniseed(fn)

   Read a miniseed file into a :class:`mth5.timeseries.RunTS` object. Uses
   `Obspy` to read the miniseed.

   :param fn: full path to the miniseed file
   :type fn: string
   :return: RunTS object
   :rtype: :class:`mth5.timeseries.RunTS`



