# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:05:33 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np

from mth5.groups.base import BaseGroup
from mth5.utils.exceptions import MTH5TableError

from mt_metadata.timeseries import (
    Survey, Station, Run, Auxiliary, Electric, Magnetic)

# =============================================================================
# Summarize standards
# =============================================================================
def summarize_metadata_standards():
    """
    Summarize metadata standards into a dictionary
    """
    summary_dict = Survey()._attr_dict
    summary_dict.add_dict(Station()._attr_dict, "station")
    summary_dict.add_dict(Run()._attr_dict, "run")
    summary_dict.add_dict(Electric()._attr_dict, "electric")
    summary_dict.add_dict(Magnetic()._attr_dict, "magnetic")
    summary_dict.add_dict(Auxiliary()._attr_dict, "auxiliary")
    
    return summary_dict
# =============================================================================
# Standards Group
# =============================================================================


class StandardsGroup(BaseGroup):
    """
    The StandardsGroup is a convenience group that stores the metadata
    standards that were used to make the current file.  This is to help a
    user understand the metadata directly from the file and not have to look
    up documentation that might not be updated.

    The metadata standards are stored in the summary table
    ``/Survey/Standards/summary``

    >>> standards = mth5_obj.standards_group
    >>> standards.summary_table
    index | attribute | type | required | style | units | description |
    options  |  alias |  example
    --------------------------------------------------------------------------

    """

    def __init__(self, group, **kwargs):

        super().__init__(group, **kwargs)

        self._defaults_summary_attrs = {
            "name": "summary",
            "max_shape": (500,),
            "dtype": np.dtype(
                [
                    ("attribute", "S72"),
                    ("type", "S15"),
                    ("required", np.bool_),
                    ("style", "S72"),
                    ("units", "S32"),
                    ("description", "S300"),
                    ("options", "S150"),
                    ("alias", "S72"),
                    ("example", "S72"),
                ]
            ),
        }

    def get_attribute_information(self, attribute_name):
        """
        get information about an attribute

        The attribute name should be in the summary table.

        :param attribute_name: attribute name
        :type attribute_name: string
        :return: prints a description of the attribute
        :raises MTH5TableError:  if attribute is not found

        >>> standars = mth5_obj.standards_group
        >>> standards.get_attribute_information('survey.release_license')
        survey.release_license
        --------------------------
                type          : string
                required      : True
                style         : controlled vocabulary
                units         :
                description   : How the data can be used. The options are based on
                         Creative Commons licenses. For details visit
                         https://creativecommons.org/licenses/
                options       : CC-0,CC-BY,CC-BY-SA,CC-BY-ND,CC-BY-NC-SA,CC-BY-NC-ND
                alias         :
                example       : CC-0

        """
        find = self.summary_table.locate("attribute", attribute_name)
        if len(find) == 0:
            msg = f"Could not find {attribute_name} in standards."
            self.logger.error(msg)
            raise MTH5TableError(msg)

        meta_item = self.summary_table.array[find]
        lines = ["", attribute_name, "-" * (len(attribute_name) + 4)]
        for name, value in zip(meta_item.dtype.names[1:], meta_item.item()[1:]):
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode()
            lines.append("\t{0:<14} {1}".format(name + ":", value))

        print("\n".join(lines))

    def summary_table_from_dict(self, summary_dict):
        """
        Fill summary table from a dictionary that summarizes the metadata
        for the entire survey.

        :param summary_dict: Flattened dictionary of all metadata standards
                             within the survey.
        :type summary_dict: dictionary

        """

        for key, v_dict in summary_dict.items():
            key_list = [key]
            for dkey in self.summary_table.dtype.names[1:]:
                value = v_dict[dkey]

                if isinstance(value, list):
                    if len(value) == 0:
                        value = ""

                    else:
                        value = ",".join(["{0}".format(ii) for ii in value])
                if value is None:
                    value = ""

                key_list.append(value)

            key_list = np.array([tuple(key_list)], self.summary_table.dtype)
            index = self.summary_table.add_row(key_list)

        self.logger.debug(f"Added {index} rows to Standards Group")

    def initialize_group(self):
        """
        Initialize the group by making a summary table that summarizes
        the metadata standards used to describe the data.

        Also, write generic metadata information.

        """
        self.initialize_summary_table()
        self.summary_table_from_dict(summarize_metadata_standards())

        self.write_metadata()
