mth5.io.miniseed.miniseed
=========================

.. py:module:: mth5.io.miniseed.miniseed

.. autoapi-nested-parse::

   Created on Wed Sep 30 10:20:12 2020

   :author: Jared Peacock

   :license: MIT



Functions
---------

.. autoapisummary::

   mth5.io.miniseed.miniseed.read_miniseed


Module Contents
---------------

.. py:function:: read_miniseed(fn)

   Read a miniseed file into a :class:`mth5.timeseries.RunTS` object. Uses
   `Obspy` to read the miniseed.

   :param fn: full path to the miniseed file
   :type fn: string
   :return: RunTS object
   :rtype: :class:`mth5.timeseries.RunTS`



