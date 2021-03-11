# -*- coding: utf-8 -*-
"""
This module provides tools to convert MT metadata to a StationXML file.

Created on Tue Jun  9 19:53:32 2020

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
from copy import deepcopy

from mth5.utils.fdsn_tools import make_channel_code, get_location_code
from mth5.utils.mth5_logger import setup_logger
from mt_metadata import timeseries as metadata

from obspy.core import inventory
from obspy.core.util import AttribDict

logger = setup_logger(__name__)
# =============================================================================
# Translate between metadata and inventory: mapping dictionaries
# =============================================================================
def flip_dict(original_dict):
    """
    Flip keys and values of the dictionary
    
    :param original_dict: DESCRIPTION
    :type original_dict: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    flipped_dict = {}

    for k, v in original_dict.items():
        if v in [None, "special"]:
            continue
        if k in [None]:
            continue
        if isinstance(v, (list, tuple)):
            # bit of a hack, needs to be more unique.
            for value in v:
                flipped_dict[value] = k
        else:
            flipped_dict[str(v)] = k

    return flipped_dict


release_dict = {
    "CC-0": "open",
    "CC-BY": "partial",
    "CC-BY-SA": "partial",
    "CC-BY-ND": "partial",
    "CC-BY-NC-SA": "partial",
    "CC-BY-NC-NC": "closed",
    None: "open",
}

base_translator = {
    "alternate_code": None,
    "code": None,
    "comments": None,
    "data_availability": None,
    "description": None,
    "historical_code": None,
    "identifiers": None,
    "restricted_status": None,
    "source_id": None,
}

### MT Survey to StationXML Network
class XMLNetworkMTSurvey:
    """
    translate back and forth between StationXML Network and MT Survey
    """

    def __init__(self):
        self.network_translator = deepcopy(base_translator)
        self.network_translator.update(
            {
                "description": "summary",
                "comments": "comments",
                "start_date": "time_period.start",
                "end_date": "time_period.end",
                "restricted_status": "release_license",
                "operators": "special",
                "code": "archive_network",
                "alternate_code": "project",
                "identifiers": ["citation_dataset.doi", "citation_journal.doi"],
            }
        )

        ### StationXML to MT Survey
        self.mt_survey_translator = flip_dict(self.network_translator)
        self.mt_survey_translator["project_lead"] = "operator"
        self.mt_survey_translator["name"] = "alternate_code"
        self.mt_survey_translator["fdsn.network"] = "code"

    def network_to_survey(self, network):
        """
        Translate a StationXML Network object to MT Survey object
        
        :param network: StationXML network element
        :type network: :class:`obspy.core.inventory.Network`
        
        """

        mt_survey = metadata.Survey()
        doi_count = 0

        for mth5_key, sxml_key in self.mt_survey_translator.items():
            if mth5_key == "project_lead":
                # only allow one person
                try:
                    inv_person = network.operators[0].contacts[0]
                    mt_survey.set_attr_from_name(
                        "project_lead.author", inv_person.names[0]
                    )
                    mt_survey.set_attr_from_name(
                        "project_lead.email", inv_person.emails[0]
                    )
                    mt_survey.set_attr_from_name(
                        "project_lead.organization", inv_person.agencies[0]
                    )
                except IndexError:
                    pass

                # is this redudant?
                try:
                    mt_survey.set_attr_from_name(
                        "project_lead.organization", network.operators[0].agencies[0],
                    )
                except IndexError:
                    pass
            elif ".doi" in mth5_key:
                try:
                    mt_survey.set_attr_from_name(
                        mth5_key, network.identifiers[doi_count]
                    )
                    doi_count += 1
                except IndexError:
                    pass

            else:
                value = getattr(network, sxml_key)
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    for k, v in zip(mth5_key, value):
                        mt_survey.set_attr_from_name(k, v)
                else:
                    if sxml_key == "restricted_status":
                        value = flip_dict(release_dict)[value]
                    if sxml_key in ["start_date", "end_date"]:
                        value = value.isoformat()

                mt_survey.set_attr_from_name(mth5_key, value)

        return mt_survey


### MT Station to StationXML Station
station_translator = deepcopy(base_translator)
station_translator.update(
    {
        "alternate_code": None,
        "channels": None,
        "code": None,
        "comments": "provenance.comments",
        "creation_date": "time_period.start",
        "data_availability": None,
        "description": "comments",
        "elevation": "location.elevation",
        "end_date": "time_period.end",
        "equipments": None,
        "external_references": None,
        "geology": None,
        "identifiers": None,
        "latitude": "location.latitude",
        "longitude": "location.longitude",
        "operators": "special",
        "site": "special",
        "start_date": "time_period.start",
        "termination_date": "time_period.end",
        "vault": None,
        "water_level": None,
    }
)

mt_station_translator = flip_dict(station_translator)


### MT Channel to StationXML Channel
channel_translator = deepcopy(base_translator)
channel_translator.update(
    {
        "azimuth": "measurement_azimuth",
        "calibration_units": "units",
        "calibration_units_description": None,
        "comments": "comments",
        "clock_drift_in_seconds_per_sample": None,
        "data_logger": "run.special",
        "description": None,
        "dip": "measurement_tilt",
        "end_date": "time_period.end",
        "equipments": None,
        "pre_amplifier": None,
        "response": None,
        "sample_rate": "sample_rate",
        "sample_rate_ratio_number_samples": None,
        "sample_rate_ratio_number_seconds": None,
        "sensor": "special",
        "start_date": "time_period.start",
        "types": "special",
        "water_level": None,
    }
)


def add_custom_element(
    obj, custom_name, custom_value, units=None, attribs=None, namespace="MT"
):
    """
    Add a custom MT element to Obspy Inventory object
    
    :param obj: :class:`~obspy.core.inventory.Inventory` object that will 
                have the element added
    :type obj: :class:`~obspy.core.inventory.Inventory`
    
    :param custom_key: name of custom element, if the key has a '.' it will
                       be recursively split to assure proper nesting.
    :type custom_key: str
    
    :param custom_value: value of custom element
    :type custom_value: [ int | float | string ]
    
    :Example: ::
        
        >>> from obspy.core import inventory
        >>> from obspy.util import AttribDict()
        >>> channel_01 = inventory.Channel('SQE', "", 39.0, -112.0, 150, 0,
        ...                                azimuth=90,
        ...                                sample_rate=256, dip=0, 
        ...                                types=['ELECTRIC POTENTIAL'])
        >>> # add custom element
        >>> channel_01.extra = AttribDict({'namespace':'MT'})
        >>> channel_01.extra.FieldNotes = AttribDict({'namespace':'MT'})
        >>> channel_01.extra.FieldNotes.value = AttribDict({'namespace':'MT'})
        >>> channel_01.extra.FieldNotes = add_custom_element(
        >>>...                                    channel_01.extra.FieldNotes,
        >>>...                                    'ContactResistanceA',
        >>>...                                    1.2,
        >>>...                                    units='kOhm')

    """

    if custom_value is None:
        return

    if "." in custom_name:
        custom_category, custom_name = custom_name.split(".", 1)
        if not hasattr(obj, custom_category):
            obj[custom_category] = AttribDict(
                {"namespace": namespace, "value": AttribDict()}
            )
        add_custom_element(
            obj[custom_category].value,
            custom_name,
            custom_value,
            units=units,
            attribs=attribs,
            namespace=namespace,
        )
    else:
        obj[custom_name] = AttribDict({"namespace": namespace})
        obj[custom_name].value = custom_value
        if units:
            assert isinstance(units, str), "Units must be a string"
            obj[custom_name].attrib = {"units": units.upper()}
        if attribs:
            if not isinstance(obj[custom_name].attrib, dict):
                obj[custom_name].attrib = {}
            for key, value in attribs.items():
                obj[custom_name].attrib[key] = value


# =============================================================================
# Translate between metadata and inventory: Survey --> Network
# =============================================================================
def mt_survey_to_inventory_network(survey_obj, namespace="MT"):
    """
    Translate MT survey metadata to inventory Network in StationXML
    
    Metadata that does not fit under StationXML schema is added as extra.
    
    :param survey_obj: MT survey metadata
    :type survey_obj: :class:`~mth5.metadata.Survey`
    :return: DESCRIPTION
    :rtype: TYPE

    """
    network_obj = inventory.Network(
        survey_obj.get_attr_from_name(network_translator["code"])
    )

    used_list = [
        "northwest_corner.latitude",
        "northwest_corner.longitude",
        "southeast_corner.latitude",
        "southeast_corner.longitude",
        "time_period.start_date",
        "time_period.end_date",
        "archive_id",
        "country",
        "datum",
        "geographic_name",
        "name",
        "hdf5_reference",
        "mth5_type",
    ]
    for inv_key, mth5_key in network_translator.items():
        if mth5_key is None:
            msg = "cannot currently map mth5.survey to network.{0}".format(inv_key)
            logger.debug(msg)
            continue
        if inv_key == "operators":
            operator = inventory.Operator(agency=[survey_obj.project_lead.organization])
            person = inventory.Person(
                names=[survey_obj.project_lead.author],
                emails=[survey_obj.project_lead.email],
            )
            operator.contacts = [person]
            network_obj.operators = [operator]
            used_list.append("project_lead.author")
            used_list.append("project_lead.email")
            used_list.append("project_lead.organization")

        elif inv_key == "comments":
            if survey_obj.comments is not None:
                comment = inventory.Comment(survey_obj.comments, id=0)
                network_obj.comments.append(comment)
        elif inv_key == "restricted_status":
            network_obj.restricted_status = release_dict[survey_obj.release_license]
        elif inv_key == "identifiers":
            for s_key in mth5_key:
                doi = survey_obj.get_attr_from_name(s_key)
                network_obj.identifiers.append(f"doi: {doi}")
                used_list.append(s_key)

        else:
            setattr(network_obj, inv_key, survey_obj.get_attr_from_name(mth5_key))
        used_list.append(mth5_key)

    # add any extra metadata that does not fit with StationXML schema
    network_obj.extra = AttribDict()
    # network_obj.extra.MT = AttribDict({'namespace': namespace,
    #                                    'value':AttribDict()})

    for mt_key in survey_obj.get_attribute_list():
        if not mt_key in used_list:
            add_custom_element(
                network_obj.extra,
                mt_key,
                survey_obj.get_attr_from_name(mt_key),
                units=survey_obj._attr_dict[mt_key]["units"],
                namespace=namespace,
            )

    return network_obj


# =============================================================================
# Translate between metadata and inventory: Station
# =============================================================================
def mt_station_to_inventory_station(station_obj, namespace="MT"):
    """
    Translate MT station metadata to inventory station
    
    Metadata that does not fit under StationXML schema is added as extra.
    
    :param station_obj: MT station metadata
    :type station_obj: :class:`~mth5.metadata.Station`

    :return: StationXML Station element
    :rtype: :class:`~obspy.core.inventory.Station`

    """
    inv_station = inventory.Station(
        station_obj.archive_id,
        station_obj.location.latitude,
        station_obj.location.longitude,
        station_obj.location.elevation,
    )

    used_list = [
        "channels_recorded",
        "time_period.start",
        "time_period.end",
        "location.latitude",
        "location.longitude",
        "location.elevation",
        "archive_id",
        "channel_layout",
        "hdf5_reference",
        "mth5_type",
    ]
    for inv_key, mth5_key in station_translator.items():
        if mth5_key is None:
            msg = "cannot currently map mth5.station to inventory.station.{0}".format(
                inv_key
            )
            logger.debug(msg)
            continue
        if inv_key == "operators":
            if station_obj.acquired_by.author is not None:
                operator = inventory.Operator(
                    agency=[station_obj.acquired_by.organization]
                )
                person = inventory.Person(names=[station_obj.acquired_by.author])
                operator.contacts = [person]
                inv_station.operators = [operator]
                used_list.append("acquired_by.author")
                used_list.append("acquired_by.organization")
        elif inv_key == "site":
            inv_station.site.description = station_obj.geographic_name
            inv_station.site.name = station_obj.id
            used_list.append("geographic_name")
            used_list.append("id")
        elif inv_key == "comments":
            if station_obj.comments is not None:
                comment = inventory.Comment(station_obj.comments, id=0)
                inv_station.comments.append(comment)
        else:
            setattr(inv_station, inv_key, station_obj.get_attr_from_name(mth5_key))

    inv_station.extra = AttribDict({})

    # make declination entry
    dec_attrs = {
        "model": station_obj.location.declination.model,
        "comment": str(station_obj.location.declination.comments),
        "units": "degrees",
    }
    inv_station.extra.declination = AttribDict(
        {
            "namespace": namespace,
            "value": station_obj.location.declination.value,
            "attrib": dec_attrs,
        }
    )
    used_list += [
        "location.declination.model",
        "location.declination.value",
        "location.declination.comments",
    ]

    # make data type entry in comments
    inv_station.comments.append(
        inventory.Comment(station_obj.data_type, subject="MT data type")
    )
    used_list.append("data_type")

    # make submitter entry in comments
    submitter = ", ".join(
        [
            "creation: {0}".format(station_obj.provenance.creation_time),
            "software: {0}".format(station_obj.provenance.software.name),
            "version: {0}".format(station_obj.provenance.software.version),
        ]
    )

    inv_station.comments.append(
        inventory.Comment(
            submitter,
            subject="metadata creation",
            authors=[
                inventory.Person(
                    names=[station_obj.provenance.submitter.author],
                    emails=[station_obj.provenance.submitter.email],
                    agencies=[station_obj.provenance.submitter.organization],
                )
            ],
        )
    )

    # make a orientation comment
    orientation = ", ".join(
        [
            "method: {0}".format(station_obj.orientation.method),
            "reference_frame: {0}".format(station_obj.orientation.reference_frame),
        ]
    )

    inv_station.comments.append(
        inventory.Comment(orientation, subject="station orientation")
    )
    used_list += ["orientation.method", "orientation.reference_frame"]

    for mt_key in station_obj.get_attribute_list():
        if "provenance" in mt_key:
            continue
        if not mt_key in used_list:
            add_custom_element(
                inv_station.extra,
                mt_key,
                station_obj.get_attr_from_name(mt_key),
                units=station_obj._attr_dict[mt_key]["units"],
                namespace=namespace,
            )

    return inv_station


# =============================================================================
# Translate between metadata and inventory: Channel
# =============================================================================
def mt_channel_to_inventory_channel(channel_obj, run_obj, namespace):
    """
    
    Translate MT channel metadata to inventory channel
    
    Metadata that does not fit under StationXML schema is added as extra.
    
    :param channel_obj: MT  channel metadata
    :type channel_obj: :class:`~mth5.metadata.Channel`, 
                       :class:`~mth5.metadata.Electric`, 
                       :class:`~mth5.metadata.Magnetic`,
                       :class:`~mth5.metadata.Auxiliary`, 
    :param run_obj: MT run metadata to get data logger information
    :type run_obj: :class:`~mth5.metadata.Run`
    :return: StationXML channel
    :rtype: :class:`~obspy.core.inventory.Channel`

    """
    location_code = get_location_code(channel_obj)
    channel_code = make_channel_code(channel_obj)

    is_electric = channel_obj.type in ["electric"]
    if is_electric:
        inv_channel = inventory.Channel(
            channel_code,
            location_code,
            channel_obj.positive.latitude,
            channel_obj.positive.longitude,
            channel_obj.positive.elevation,
            channel_obj.positive.elevation,
        )
    else:
        inv_channel = inventory.Channel(
            channel_code,
            location_code,
            channel_obj.location.latitude,
            channel_obj.location.longitude,
            channel_obj.location.elevation,
            channel_obj.location.elevation,
        )

    inv_channel.start_date = run_obj.time_period.start
    inv_channel.end_date = run_obj.time_period.end
    print(inv_channel.start_date)

    used_list = [
        "channels_recorded",
        "time_period.start",
        "time_period.end",
        "sample_rate",
        "location.latitude",
        "location.longitude",
        "location.elevation",
        "measurement_azimuth",
        "measurement_tilt",
        "units",
        "hdf5_reference",
        "mth5_type",
    ]
    for inv_key, mth5_key in channel_translator.items():
        if mth5_key is None:
            msg = "cannot currently map mth5.station to inventory.station.{0}".format(
                inv_key
            )
            logger.debug(msg)
            continue

        if inv_key == "data_logger":
            dl = inventory.Equipment()
            dl.manufacturer = run_obj.data_logger.manufacturer
            dl.model = run_obj.data_logger.model
            dl.serial_number = run_obj.data_logger.id
            dl.type = run_obj.data_logger.type
            inv_channel.data_logger = dl

        elif inv_key == "sensor":
            if is_electric:
                for direction in ["positive", "negative"]:
                    desc = ", ".join(
                        [
                            f"{direction} electrode",
                            "latitude: {0}".format(
                                channel_obj.get_attr_from_name(f"{direction}.latitude")
                            ),
                            "longitude: {0}".format(
                                channel_obj.get_attr_from_name(f"{direction}.longitude")
                            ),
                            "elevation: {0}".format(
                                channel_obj.get_attr_from_name(f"{direction}.elevation")
                            ),
                        ]
                    )
                    sensor = inventory.Equipment()
                    sensor.manufacturer = channel_obj.get_attr_from_name(
                        f"{direction}.manufacturer"
                    )
                    sensor.type = channel_obj.get_attr_from_name(f"{direction}.type")
                    sensor.model = channel_obj.get_attr_from_name(f"{direction}.model")
                    sensor.description = desc

                    inv_channel.equipments.append(sensor)
            else:
                sensor = inventory.Equipment()
                sensor.manufacturer = channel_obj.sensor.manufacturer
                sensor.type = channel_obj.sensor.type
                sensor.model = channel_obj.sensor.model
                sensor.serial_number = channel_obj.sensor.id
                inv_channel.sensor = sensor

        elif inv_key == "comments":
            comment = inventory.Comment(channel_obj.comments, id=0)
            inv_channel.comments.append(comment)
            used_list.append("comments")

        # obspy only allows angles (0, 360)
        elif inv_key in ["azimuth"]:
            inv_channel.azimuth = channel_obj.measurement_azimuth % 360

        elif inv_key in ["dip"]:
            inv_channel.dip = channel_obj.measurement_tilt % 360

        elif inv_key == "types":
            inv_channel.types = ["GEOPHYSICAL"]
        else:
            setattr(inv_channel, inv_key, channel_obj.get_attr_from_name(mth5_key))

    inv_channel.extra = AttribDict()
    inv_channel.extra.Magnetotellurics = AttribDict(
        {"namespace": namespace, "value": AttribDict()}
    )
    inv_channel.extra.Magnetotellurics.attrib = {
        "type": channel_obj.type,
        "component": channel_obj.component,
        "channel_number": str(channel_obj.channel_number),
    }
    used_list += ["type", "component", "channel_number"]
    # inv_channel.extra.Magnetotellurics.value =

    for mt_key in channel_obj.get_attribute_list():
        if "positive" in mt_key or "negative" in mt_key or "sensor" in mt_key:
            continue
        if not mt_key in used_list:
            add_custom_element(
                inv_channel.extra.Magnetotellurics.value,
                mt_key,
                channel_obj.get_attr_from_name(mt_key),
                units=channel_obj._attr_dict[mt_key]["units"],
                namespace=namespace,
            )

    return inv_channel


# =============================================================================
#
# =============================================================================
class MTToStationXML:
    """
    Translate MT metadata to StationXML using Obspy Inventory classes.
    
    Any metadata that does not fit under the StationXML schema will be added
    as extra metadata in the namespace MT.
    
    MT metadata is mapped into StationXML
    
    >>> # inventory mapping
    Inventory 
    ===========
      |--> Network (MT Survey)
      -------------- 
        |--> Station (MT Station)
        -------------
          |--> Channel (MT Channel + MT Run)
          -------------
              
    :Example:
        
    >>> from mth5.utils import translator
    >>> from mth import metadata
    >>> # survey_dict = metadata for survey 
    >>> mt2xml = translator.MTToStationXML()
    >>> mt_survey = metadata.Survey()
    >>> mt_survey.from_dict(survey_dict)
    >>> mt2xml.add_network(mt_survey)
    
    :Add a station from an xml file with root <station>:
        
    >>> from xml.etree import ElementTree as et
    >>> mt_station = metadata.Station()
    >>> mt_station.from_xml(et.parse("mt_station_xml_fn.xml").getroot())
    >>> mt2xml.add_station(mt_station)
    
    :Add a channel from an json files with {channel:{}} and {run:{}} format:

    >>> import json
    >>> mt_electric = metadata.Electric()
    >>> with open("electric_json_fn.json", 'r') as fid:
    >>> ... mt_electric.from_json(json.load(fid))
    >>> mt_run = metadata.Run()
    >>> with open("run_json_fn.json", 'r') as fid:
    >>> ... mt_run.from_json(json.load(fid))
    >>> mt2xml.add_channel(mt_electric, mt_run, mt_station.archive_id)
        
    """

    def __init__(self, inventory_object=None):

        self.logger = setup_logger("{0}.{1}".format(__name__, self.__class__.__name__))

        self.mt_namespace = r"http://emiw.org/xmlns/mt/1.0"
        self.namespace_map = {
            "xsi": r"http://www.w3.org/2001/XMLSchema-instance",
            "schemaLocation": "http://www.fdsn.org/xml/station/fdsn-station-1.1.xsd",
            "mt": self.mt_namespace,
        }

        if inventory_object is not None:
            if not isinstance(inventory_object, inventory.Inventory):
                msg = "Input must be obspy.Inventory object not type {0}"
                self.logger.error(msg.format(type(inventory_object)))
                raise TypeError(msg.format(type(inventory_object)))
            self.inventory_obj = inventory_object

        else:
            self.inventory_obj = inventory.Inventory(source="MT Metadata")

    def find_network_index(self, network):
        """
        locate the index of given network
        
        :param network: name of the network to look for
        :type network: 2 character string
        
        :return: index of network in inventory.networks
        :rtype: int
        
        """

        for ii, net in enumerate(self.inventory_obj.networks):
            if network == net.code:
                return ii
        return None

    def find_station_index(self, station, network=None):
        """
        locate the index of given station in 
        Inventory.networks[network].stations
        
        :param station: 5 character SEED station name
        :type station: 5 character string
        :param network: Network name, defaults to None which will use 
                        Inventory.networks[0] 
        :type network: 2 character string, optional
        :return: Index of station in Inventory.networks[network].stations
        :rtype: int

        """

        if network is not None:
            network_index = self.find_network_index(network)
        else:
            network_index = 0

        for ii, sta in enumerate(self.inventory_obj.networks[network_index].stations):
            if station == sta.code:
                return ii

        return None

    def add_network(self, mt_survey_obj):
        """
        Add a network from an MT survey object.  Will fill the appropriate
        metadata in Inventory.Network, any metadata that does not fit within
        the StationXML schema will be added as extra.
        
        :param mt_survey_obj: MT Survey metadata 
        :type mt_survey_obj: :class:`~mth5.metadata.Survey`

        """
        network_obj = mt_survey_to_inventory_network(
            mt_survey_obj, namespace=self.mt_namespace
        )

        if network_obj.code in self.inventory_obj.networks:
            msg = "Network {0} is alread in current inventory".format(network_obj.code)
            self.logger.error(msg)
            raise ValueError(msg)
        self.inventory_obj.networks.append(network_obj)

        self.logger.debug(
            "Added network {0} to inventory".format(mt_survey_obj.archive_network)
        )

    def add_station(self, mt_station_obj, network_code=None):
        """
        
        Add a station from an MT station object.  Will fill the appropriate
        metadata in Inventory.Network[network].station, any metadata that
        does not fit within the StationXML schema will be added as extra.
        
        :param mt_station_obj: MT station metadata
        :type mt_station_obj: :class:`~mth5.metadata.Station`
        :param network_code: Network code that the station belongs to. 
                             Defaults to None which will  use
                             Inventory.networks[0]
        :type network_code: 2 character code. optional

        """
        if network_code is None:
            network_code = self.inventory_obj.networks[0].code

        network_index = self.find_network_index(network_code)

        station_obj = mt_station_to_inventory_station(
            mt_station_obj, namespace=self.mt_namespace
        )

        # locate the network in the list
        self.inventory_obj.networks[network_index].stations.append(station_obj)

        msg = "Added station {0} to network {1}".format(
            mt_station_obj.archive_id, network_code
        )
        self.logger.debug(msg)

    def add_channel(self, mt_channel, mt_run, station, network_code=None):
        """
        Add a station from a MT channel and run objects. The run object is
        needed to fill the datalogger information.
        
        Will fill the appropriate metadata in 
        Inventory.Network[network].station[station], any metadata that
        does not fit within the StationXML schema will be added as extra.
        
        :param mt_channel: MT channel metadata
        :type mt_channel: :class:`~mth5.metadata.Channel`, 
                          :class:`~mth5.metadata.Electric`, 
                          :class:`~mth5.metadata.Magnetic`,
                          :class:`~mth5.metadata.Auxiliary`
        :param mt_run: MT run metadata
        :dtype mt_run: :class:`~mth5.metadata.Run`
        :param station: Station name to add the channel to
        :type station: 5 character string
        :param network_code: Network code that the station belongs to. 
                             Defaults to None which will  use
                             Inventory.networks[0]
        :type network_code: 2 character code. optional

        """

        if network_code is None:
            network_code = self.inventory_obj.networks[0].code

        network_index = self.find_network_index(network_code)
        station_index = self.find_station_index(station, network_code)

        channel_obj = mt_channel_to_inventory_channel(
            mt_channel, mt_run, namespace=self.mt_namespace
        )

        self.inventory_obj.networks[network_index].stations[
            station_index
        ].channels.append(channel_obj)

        self.logger.debug(
            "Added channel {0} with code {1} to station {2} for netowrk {3}".format(
                mt_channel.component, channel_obj.code, station, network_code
            )
        )

    def to_stationxml(self, station_xml_fn):
        """
        Write a StationXML file using Inventory.write
        
        :param station_xml_fn: Full path to StationXML file
        :type station_xml_fn: string or Path
        :return: path to StationXML
        :rtype: Path

        """
        if not isinstance(station_xml_fn, Path):
            station_xml_fn = Path(station_xml_fn)

        if station_xml_fn.exists():
            msg = "File {0} already exists and will be over written".format(
                station_xml_fn
            )
            self.logger.warning(msg)

        self.inventory_obj.write(
            station_xml_fn.as_posix(),
            format="stationxml",
            validate=True,
            nsmap=self.namespace_map,
        )
        self.logger.info("Wrote StationXML to {0}".format(station_xml_fn))

        return station_xml_fn


# =============================================================================
# Translate from stationxml to mth5
# =============================================================================
def read_comment(inv_comment):
    """
    
    :param inv_comment: DESCRIPTION
    :type inv_comment: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    a_dict = {}
    key = inv_comment.subject.strip().replace(" ", "_").lower()

    if ":" in inv_comment.value:
        a_dict[key] = {}
        a_list = inv_comment.value.split(",")
        for aa in a_list:
            k, v = [vv.strip() for vv in aa.split(":", 1)]
            a_dict[key][k] = v

    return a_dict


def inventory_network_to_mt_survey(network_obj):
    """
    Convert an inventory.Network oject to an :class:`mth5.metadata.Survey` 
    object
    
    :param network_obj: DESCRIPTION
    :type network_obj: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    mt_survey = metadata.Survey()
    doi_count = 0

    for mth5_key, sxml_key in mt_survey_translator.items():
        if mth5_key == "project_lead":
            # only allow one person
            try:
                inv_person = network_obj.operators[0].contacts[0]
                mt_survey.set_attr_from_name("project_lead.author", inv_person.names[0])
                mt_survey.set_attr_from_name("project_lead.email", inv_person.emails[0])
                mt_survey.set_attr_from_name(
                    "project_lead.organization", inv_person.agencies[0]
                )
            except IndexError:
                pass

            # is this redudant?
            try:
                mt_survey.set_attr_from_name(
                    "project_lead.organization", network_obj.operators[0].agencies[0],
                )
            except IndexError:
                pass
        elif ".doi" in mth5_key:
            try:
                mt_survey.set_attr_from_name(
                    mth5_key, network_obj.identifiers[doi_count]
                )
                doi_count += 1
            except IndexError:
                pass

        else:
            value = getattr(network_obj, sxml_key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for k, v in zip(mth5_key, value):
                    mt_survey.set_attr_from_name(k, v)
            else:
                if sxml_key == "restricted_status":
                    value = flip_dict(release_dict)[value]
                if sxml_key in ["start_date", "end_date"]:
                    value = value.isoformat()

            mt_survey.set_attr_from_name(mth5_key, value)

    return mt_survey


def inventory_station_to_mt_station(inv_station_obj):
    """
    
    :param inv_station_obj: DESCRIPTION
    :type inv_station_obj: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    mt_station = metadata.Station()

    for mth5_key, sxml_key in mt_station_translator.items():
        # print(f"MTH5 = {mth5_key}\nStationXML = {sxml_key}")
        value = getattr(inv_station_obj, sxml_key)
        if "date" in sxml_key:
            value = value.isoformat()

        if isinstance(value, inventory.Comment):
            v_dict = read_comment(value)

        mt_station.set_attr_from_name(mth5_key, value)

    return mt_station


class StationXMLToMTH5:
    """
    Translate a station XML to MT metadata standards
    
    """

    pass
