mth5.io.phoenix.readers.helpers
===============================

.. py:module:: mth5.io.phoenix.readers.helpers

.. autoapi-nested-parse::

   Helper utilities for Phoenix Geophysics reader module.

   Created on Tue Jun 20 15:51:20 2023

   @author: jpeacock



Functions
---------

.. autoapisummary::

   mth5.io.phoenix.readers.helpers.read_json_to_object


Module Contents
---------------

.. py:function:: read_json_to_object(fn: str | pathlib.Path) -> types.SimpleNamespace

   Read a JSON file directly into a SimpleNamespace object.

   :param fn: Path to the JSON file to read.
   :type fn: str or Path

   :returns: Object containing the JSON data as attributes.
   :rtype: SimpleNamespace

   :raises FileNotFoundError: If the specified file does not exist.
   :raises json.JSONDecodeError: If the file contains invalid JSON.
   :raises IOError: If there's an error reading the file.

   .. rubric:: Examples

   >>> obj = read_json_to_object("config.json")
   >>> print(obj.some_attribute)

   .. rubric:: Notes

   This function uses json.load with an object_hook to convert
   all dictionaries to SimpleNamespace objects, allowing dot
   notation access to nested JSON properties.


