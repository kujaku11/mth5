# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:41:16 2020

:copyright: 
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""
# =============================================================================
# Imports
# =============================================================================
import json
import pandas as pd
import numpy as np
import logging

from collections import OrderedDict
from collections.abc import Iterable
from operator import itemgetter

from mth5.metadata.standards.schema import validate_attribute, validate_type
from mth5.utils.exceptions import MTSchemaError
from mth5.metadata import helpers

# =============================================================================
#  Base class that everything else will inherit
# =============================================================================
class Base:
    """
    A Base class that is common to most of the Metadata objects

    Includes:
        
        * to_json
        * from_json
        * to_dict
        * from_dict
        * to_series
        * from_series
        
    """

    def __init__(self, attr_dict={}, **kwargs):

        self._attr_dict = attr_dict

        self._class_name = validate_attribute(self.__class__.__name__)

        self.logger = logging.getLogger(f"{__name__}.{self._class_name}")

        for name, value in kwargs.items():
            self.set_attr_from_name(name, value)

    def __str__(self):
        meta_dict = self.to_dict()[self._class_name.lower()]
        lines = ["{0}:".format(self._class_name)]
        for name, value in meta_dict.items():
            lines.append("\t{0} = {1}".format(name, value))
        return "\n".join(lines)

    def __repr__(self):
        return self.to_json()

    def __eq__(self, other):
        if isinstance(other, (Base, dict, str, pd.Series)):
            home_dict = self.to_dict()[self._class_name]
            if isinstance(other, Base):
                other_dict = other.to_dict()[self._class_name]
            elif isinstance(other, dict):
                other_dict = other
            elif isinstance(other, str):
                other_dict = OrderedDict(
                    sorted(json.loads(other).items(), key=itemgetter(0))
                )
            elif isinstance(other, pd.Series):
                other_dict = OrderedDict(
                    sorted(other.to_dict().items(), key=itemgetter(0))
                )
            if other_dict == home_dict:
                return True
            else:
                for key, value in home_dict.items():
                    try:
                        other_value = other_dict[key]
                        if value != other_value:
                            msg = f"{key}: {value} != {other_value}"
                            self.logger.info(msg)
                    except KeyError:
                        msg = "Cannot find {0} in other".format(key)
                        self.logger.info(msg)

                return False
        raise ValueError(f"Cannot compare {self._class_name} with {type(other)}")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.get_attribute_list())

    def get_attribute_list(self):
        """
        return a list of the attributes 
        """

        return sorted(list(self._attr_dict.keys()))

    def attribute_information(self, name=None):
        """
        return a descriptive string of the attribute if none returns for all
    
        :param key: DESCRIPTION, defaults to None
        :type key: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if name:
            try:
                v_dict = OrderedDict(
                    sorted(self._attr_dict[name].items(), key=itemgetter(0))
                )
            except KeyError as error:
                msg = "{0} not attribute {1} found".format(error, name)
                self.logger.error(msg)
                raise MTSchemaError(msg)

            lines = ["{0}:".format(name)]
            for key, value in v_dict.items():
                lines.append("\t{0}: {1}".format(key, value))
        else:
            lines = []
            for name, v_dict in self._attr_dict.items():
                lines.append("{0}:".format(name))
                v_dict = OrderedDict(sorted(v_dict.items(), key=itemgetter(0)))
                for key, value in v_dict.items():
                    lines.append("\t{0}: {1}".format(key, value))
                lines.append("=" * 50)

        print("\n".join(lines))

    def _validate_name(self, name):
        """
        validate the name to conform to the standards
        name must be:
            * all lower case {a-z; 1-9}
            * must start with a letter
            * categories are separated by '.'
            * words separated by '_'

        {object}.{name_name}

        '/' will be replaced with '.'
        converted to all lower case

        :param name: name name
        :type name: string
        :return: valid name name
        :rtype: string

        """
        return validate_attribute(name)

    def _validate_type(self, value, v_type, style=None):
        """
        validate type from standards
        
        """
        # return if the value is None, this may need to change in the future
        # if an empty list or something else should be returned
        if not isinstance(value, (list, tuple, np.ndarray)):
            if value in [None, "None", "none", "unknown"]:
                return None
        # hack to get around h5py reference types, in the future will need
        # a more robust test.
        if v_type == "h5py_reference":
            return value

        # return value if the value type is not defined.
        if v_type is None:
            msg = (
                "standards data type is unknown, if you want to "
                + "propogate this attribute using to_dict, to_json or "
                + "to_series, you need to add attribute description using "
                + "class function add_base_attribute."
                + "Example: \n\t>>> Run.add_base_attribute(new, 10, "
                + '{"type":float, "required": True, "units": None, '
                + '"style": number})'
            )
            self.logger.info(msg)
            return value

        # if not a python type but a string organize into a dictionary
        if not isinstance(v_type, type) and isinstance(v_type, str):
            type_dict = {"string": str, "integer": int, "float": float, "boolean": bool}
            v_type = type_dict[validate_type(v_type)]
        else:
            msg = "v_type must be a string or type not {0}".format(v_type)

        # check style for a list
        if isinstance(value, v_type):
            if style:
                if v_type is str and "list" in style:
                    value = value.replace("[", "").replace("]", "").split(",")
                    value = [ss.strip() for ss in value]
            return value

        # if value is not of v_type
        else:
            msg = "value={0} must be {1} not {2}"
            info = "converting {0} to {1}"
            # if the value is a string, convert to appropriate type
            if isinstance(value, str):
                if v_type is int:
                    try:
                        self.logger.debug(info.format(type(value), v_type))
                        return int(value)
                    except ValueError as error:
                        self.logger.exception(error)
                        raise MTSchemaError(msg.format(value, v_type, type(value)))
                elif v_type is float:
                    try:
                        self.logger.debug(info.format(type(value), v_type))
                        return float(value)
                    except ValueError as error:
                        self.logger.exception(error)
                        raise MTSchemaError(msg.format(value, v_type, type(value)))
                elif v_type is bool:
                    if value.lower() in ["false", "0"]:
                        self.logger.debug(info.format(value, False))
                        return False
                    elif value.lower() in ["true", "1"]:
                        self.logger.debug(info.format(value, True))
                        return True
                    else:
                        self.logger.exception(msg.format(value, v_type, type(value)))
                        raise MTSchemaError(msg.format(value, v_type, type(value)))
                elif v_type is str:
                    return value

            # if a number convert to appropriate type
            elif isinstance(value, (int, np.int_)):
                if v_type is float:
                    self.logger.debug(info.format(type(value), v_type))
                    return float(value)
                elif v_type is str:
                    self.logger.debug(info.format(type(value), v_type))
                    return "{0:.0f}".format(value)
                return int(value)

            # if a number convert to appropriate type
            elif isinstance(value, (float, np.float_)):
                if v_type is int:
                    self.logger.debug(info.format(type(value), v_type))
                    return int(value)
                elif v_type is str:
                    self.logger.debug(info.format(type(value), v_type))
                    return "{0}".format(value)
                return float(value)

            # if a list convert to appropriate entries to given type
            elif isinstance(value, Iterable):
                if v_type is str:
                    if isinstance(value, np.ndarray):
                        value = value.astype(np.unicode_)
                    value = [f"{v}" for v in value]
                elif v_type is int:
                    value = [int(float(v)) for v in value]
                elif v_type is float:
                    value = [float(v) for v in value]
                elif v_type is bool:
                    value_list = []
                    for v in value:
                        if v in [True, "true", "True", "TRUE"]:
                            value_list.append(True)
                        elif v in [False, "false", "False", "FALSE"]:
                            value_list.append(False)
                    value = value_list
                return value

            elif isinstance(value, (np.bool_)):
                return bool(value)

            else:
                self.logger.exception(msg.format(value, v_type, type(value)))
                raise MTSchemaError(msg.format(value, v_type, type(value)))
        return None

    def _validate_option(self, name, option_list):
        """
        validate the given attribute name agains possible options and check
        for aliases
        
        :param name: DESCRIPTION
        :type name: TYPE
        :param option_list: DESCRIPTION
        :type option_list: TYPE
        :param alias_list: DESCRIPTION
        :type alias_list: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if name is None:
            return True, False, None

        options = [ss.lower() for ss in option_list]
        other_possible = False
        if "other" in options:
            other_possible = True
        if name.lower() in options:
            return True, other_possible, None
        elif name.lower() not in options and other_possible:
            msg = (
                "{0} not found in options list {1}, but other options"
                + " are allowed.  Allowing {2} to be set to {0}."
            )
            return True, other_possible, msg

        return False, other_possible, "{0} not found in options list {1}"

    def __setattr__(self, name, value):
        """
        set attribute based on metadata standards

        """
        # skip these attribute because they are validated in the property
        # setter.
        skip_list = [
            "latitude",
            "longitude",
            "elevation",
            "start_date",
            "end_date",
            "start",
            "end",
            "name",
            "applied",
            "logger",
        ]

        if hasattr(self, "_attr_dict"):
            if name[0] != "_":
                if not name in skip_list:
                    self.logger.debug("Setting {0} to {1}".format(name, value))
                    v_dict = self._attr_dict[name]
                    v_type = self._get_standard_type(name)
                    value = self._validate_type(value, v_type, v_dict["style"])
                    # check options
                    if v_dict["style"] == "controlled vocabulary":
                        options = v_dict["options"]
                        accept, other, msg = self._validate_option(value, options)
                        if not accept:
                            self.logger.error(msg.format(value, options))
                            raise MTSchemaError(msg.format(value, options))
                        if other and not accept:
                            self.logger.warning(msg.format(value, options, name))

        super().__setattr__(name, value)

    def _get_standard_type(self, name):
        """
        helper function to get the standard type for the given name
        """
        name = self._validate_name(name)
        try:
            standards = self._attr_dict[name]
            return standards["type"]
        except KeyError:
            if name[0] != "_":
                msg = (
                    "{0} is not defined in the standards. "
                    + " Should add attribute information with "
                    + "add_base_attribute if the attribute is going to "
                    + "propogate via to_dict, to_json, to_series"
                )
                self.logger.info(msg.format(name))
            return None

    def get_attr_from_name(self, name):
        """
        Access attribute from the given name.

        The name can contain the name of an object which must be separated
        by a '.' for  e.g. {object_name}.{name} --> location.latitude

        ..note:: this is a helper function for names with '.' in the name for
                 easier getting when reading from dictionary.

        :param name: name of attribute to get.
        :type name: string
        :return: attribute value
        :rtype: type is defined by the attribute name

        :Example:

        >>> b = Base(**{'category.test_attr':10})
        >>> b.get_attr_from_name('category.test_attr')
        10

        """
        name = self._validate_name(name)
        v_type = self._get_standard_type(name)

        if "." in name:
            value = helpers.recursive_split_getattr(self, name)
        else:
            value = getattr(self, name)

        return self._validate_type(value, v_type)

    def set_attr_from_name(self, name, value):
        """
        Helper function to set attribute from the given name.

        The name can contain the name of an object which must be separated
        by a '.' for  e.g. {object_name}.{name} --> location.latitude

        ..note:: this is a helper function for names with '.' in the name for
                 easier getting when reading from dictionary.

        :param name: name of attribute to get.
        :type name: string
        :param value: attribute value
        :type value: type is defined by the attribute name

        :Example: 

        >>> b = Base(**{'category.test_attr':10})
        >>> b.set_attr_from_name('category.test_attr', '10')
        >>> print(b.category.test_attr)
        '10'
        """
        if "." in name:
            try:
                helpers.recursive_split_setattr(self, name, value)
            except AttributeError as error:
                msg = (
                    "{0} is not in the current standards.  "
                    + "To properly add the attribute use "
                    + "add_base_attribute."
                )

                self.logger.error(msg.format(name))
                self.logger.exception(error)
                raise AttributeError(error)
        else:
            setattr(self, name, value)

    def add_base_attribute(self, name, value, value_dict):
        """
        Add an attribute to _attr_dict so it will be included in the
        output dictionary
        
        :param name: name of attribute
        :type name: string
        
        :param value: value of the new attribute
        :type value: described in value_dict
        
        :param value_dict: dictionary describing the attribute, must have keys
            ['type', 'required', 'style', 'units', 'alias', 'description',
             'options', 'example']
        :type name: string
    
        * type --> the data type [ str | int | float | bool ]
        * required --> required in the standards [ True | False ]
        * style --> style of the string
        * units --> units of the attribute, must be a string
        * alias --> other possible names for the attribute
        * options --> if only a few options are accepted, separated by | or 
          comma.b [ option_01 | option_02 | other ]. 'other' means other options 
          available but not yet defined.
        * example --> an example of the attribute
        
        :Example:
            
        >>> extra = {'type': str,
        >>> ...      'style': 'controlled vocabulary',
        >>> ...      'required': False,
        >>> ...      'units': celsius,
        >>> ...      'description': 'local temperature',
        >>> ...      'alias': ['temp'],
        >>> ...      'options': [ 'ambient', 'air', 'other'],
        >>> ...      'example': 'ambient'}
        >>> r = Run()
        >>> r.add_base_attribute('temperature', 'ambient', extra)

        """
        name = self._validate_name(name)
        self._attr_dict.update({name: value_dict})
        self.set_attr_from_name(name, value)
        self.logger.debug("Added {0} to _attr_dict with {1}".format(name, value_dict))
        self.logger.debug(
            "set {0} to {1} as type {2}".format(name, value, value_dict["type"])
        )

    def to_dict(self, nested=False, single=False, required=True):
        """
        make a dictionary from attributes, makes dictionary from _attr_list.
        
        :param nested: make the returned dictionary nested
        :type nested: [ True | False ] , default is False
        
        :param single: return just metadata dictionary -> meta_dict[class_name]
        :type single: [ True | False ], default is False
        
        :param required: return just the required elements and any elements with
                         non-None values
        
        """
        meta_dict = {}
        for name in list(self._attr_dict.keys()):
            try:
                value = self.get_attr_from_name(name)
            except AttributeError as error:
                msg = "{0}: setting {1} to None.  ".format(
                    error, name
                ) + "Try setting {0} to the desired value".format(name)
                self.logger.debug(msg)
                value = None

            if required:
                if value is not None or self._attr_dict[name]["required"]:
                    meta_dict[name] = value
            else:
                meta_dict[name] = value

        if nested:
            meta_dict = helpers.structure_dict(meta_dict)

        meta_dict = {
            self._class_name.lower(): OrderedDict(
                sorted(meta_dict.items(), key=itemgetter(0))
            )
        }

        if single:
            meta_dict = meta_dict[list(meta_dict.keys())[0]]

        return meta_dict

    def from_dict(self, meta_dict):
        """
        fill attributes from a dictionary
        
        :param meta_dict: dictionary with keys equal to metadata.
        :type meta_dict: dictionary
        
        """
        if not isinstance(meta_dict, (dict, OrderedDict)):
            msg = "Input must be a dictionary not {0}".format(type(meta_dict))
            self.logger.error(msg)
            raise MTSchemaError(msg)

        keys = list(meta_dict.keys())
        if len(keys) == 1:
            class_name = keys[0]
            if class_name.lower() != self._class_name.lower():
                msg = (
                    "name of input dictionary is not the same as class type "
                    "input = {0}, class type = {1}".format(class_name, self._class_name)
                )
                self.logger.debug(msg)
            meta_dict = helpers.flatten_dict(meta_dict[class_name])
        else:
            self.logger.debug(
                f"Assuming input dictionary is of type {self._class_name}"
            )
            meta_dict = helpers.flatten_dict(meta_dict)

        # set attributes by key.
        for name, value in meta_dict.items():
            self.set_attr_from_name(name, value)

    def to_json(self, nested=False, indent=" " * 4, required=True):
        """
        Write a json string from a given object, taking into account other
        class objects contained within the given object.
        
        :param nested: make the returned json nested
        :type nested: [ True | False ] , default is False
        
        """

        return json.dumps(
            self.to_dict(nested=nested, required=required),
            cls=helpers.NumpyEncoder,
            indent=indent,
        )

    def from_json(self, json_str):
        """
        read in a json string and update attributes of an object

        :param json_str: json string
        :type json_str: string

        """
        if not isinstance(json_str, str):
            msg = "Input must be valid JSON string not {0}".format(type(json_str))
            self.logger.error(msg)
            raise MTSchemaError(msg)

        self.from_dict(json.loads(json_str))

    def from_series(self, pd_series):
        """
        Fill attributes from a Pandas series
        
        .. note:: Currently, the series must be single layered with key names
                  separated by dots. (location.latitude)
        
        :param pd_series: Series containing metadata information
        :type pd_series: pandas.Series
        
        ..todo:: Force types in series
        
        """
        if not isinstance(pd_series, pd.Series):
            msg = "Input must be a Pandas.Series not type {0}".format(type(pd_series))
            self.logger.error(msg)
            MTSchemaError(msg)
        for key, value in pd_series.iteritems():
            self.set_attr_from_name(key, value)

    def to_series(self, required=True):
        """
        Convert attribute list to a pandas.Series
        
        .. note:: this is a flattened version of the metadata
        
        :return: pandas.Series
        :rtype: pandas.Series

        """

        return pd.Series(self.to_dict(single=True, required=required))

    def to_xml(self, string=False, required=True):
        """
        make an xml element for the attribute that will add types and 
        units.  
        
        :param string: output a string instead of an XML element
        :type string: [ True | False ], default is False
        
        :return: XML element or string

        """
        element = helpers.dict_to_xml(
            self.to_dict(nested=True, required=required), self._attr_dict
        )
        if not string:
            return element
        else:
            return helpers.element_to_string(element)

    def from_xml(self, xml_element):
        """
        
        :param xml_element: XML element
        :type xml_element: etree.Element
        
        :return: Fills attributes accordingly

        """

        self.from_dict(helpers.element_to_dict(xml_element))
