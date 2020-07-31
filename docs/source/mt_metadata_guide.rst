====================================================
A Standard for Exchangeable Magnetotelluric Metadata
====================================================

:Author: Working Group for Data Handling and Software - PASSCAL Magnetotelluric Program
:Date:   **Version 0.0.16 – July 2020**\  [1]_

Introduction
============

Researchers using magnetotelluric (MT) methods lack a standardized
format for storing time series data and metadata. Commercially available
MT instruments produce data in formats that range from proprietary
binary to ASCII, whereas recent datasets from the U.S. MT community have
utilized institutional formats or heavily adapted formats like miniSEED.
In many cases, the available metadata for MT time series are incomplete
and loosely standardized; and overall, these datasets are not "user
friendly". This lack of a standardized resource impedes the exchange and
broader use of these data beyond a small community of specialists.

The `IRIS PASSCAL MT
facility <https://www.iris.edu/hq/programs/passcal/magnetotelluric_instrumentation>`__
maintains a pool of MT instruments that are freely available to U.S.
Principal Investigators (PIs). Datasets collected with these instruments
are subject to data sharing requirements, and an IRIS `working
group <https://www.iris.edu/hq/about_iris/governance/mt_soft>`__ advises
the development of sustainable data formats and workflows for this
facility. Following in the spirit of the standard created for `MT
transfer
function <https://library.seg.org/doi/10.1190/geo2018-0679.1>`__
datasets, this document outlines a new metadata standard for level
0,1,and 2 MT time series data (`Data
Levels <https://earthdata.nasa.gov/collaborate/open-data-services-and-software/data-information-policy/data-levels>`__).
Following community approval of these standards, MTH5 (an HDF5 MT
specific format) will be developed later in 2020.

The Python 3 module written for these standards and MTH5 is being
developed at https://github.com/kujaku11/MTarchive/tree/tables.

General Structure
=================

The metadata for a full MT dataset are structured to cover details from
single channel time series to a full survey. For simplicity, each of the
different scales of an MT survey and measurements have been categorized
starting from largest to smallest (Figure `1 <#fig:example>`__). These
categories are: ``Survey``, ``Station``, ``Run``, ``DataLogger``,
``Electric Channel``, ``Magnetic Channel``, and ``Auxiliary Channel``.
Each category is described in subsequent sections. Required keywords are
labeled as and suggested keywords are labeled as . A user should use as
much of the suggested metadata as possible for a full description of the
data.

.. figure:: images/example_mt_file_structure.png

   Schematic of a MT time series file structure with appropriate
   metadata. The top level is the *Survey* that contains general
   information about who, what, when, where, and how the data were
   collected. Underneath *Survey* are the *Station* and *Filter*.
   *Filter* contains information about different filters that need to be
   applied to the raw data to get appropriate units and calibrated
   measurements. Underneath *Station* are *Run*, which contain data that
   were collected at a single sampling rate with common start and end
   time at a single station. Finally, *Channel* describes each channel
   of data collected and can be an *Auxiliary*, *Electric*, or
   *Magnetic*. Metadata is attributed based on the type of data
   collected in the channel.

Metadata Keyword Format
-----------------------

| The metadata key names should be self-explanatory and are structured
  as follows:
| ``{category}.{name}``, or can be nested
  ``{category1}.{categroy2}.{name}`` where:

-  ``category`` refers to a metadata category or level that has common
   parameters, such as ``location``, which will have a latitude,
   longitude, and elevation :math:`\longrightarrow`
   ``location.latitude``, ``location.longitude``, and
   ``location.elevation``. These can be nested, for example,
   ``station.location.latitude``

-  ``name`` is a descriptive name, where words should be separated by an
   underscore. Note that only whole words should be used and
   abbreviations should be avoided, e.g. ``data_quality``.

A ‘.’ represents the separator between different categories. The
metadata can be stored in many different forms. Common forms are XML or
JSON formats. See examples below for various ways to represent the
metadata.

Formatting Standards
--------------------

Specific and required formatting standards for location, time and date,
and angles are defined below and should be adhered to.

Time and Date Format
~~~~~~~~~~~~~~~~~~~~

All time and dates are given as an ISO formatted date-time String in the
UTC time zone. The ISO Date Time format is
``YYYY-MM-DDThh:mm:ss.ms+00:00``, where the UTC time zone is represented
by ``+00:00``. UTC can also be denoted by ``Z`` at the end of the
date-time string ``YYYY-MM-DDThh:mm:ss.msZ``. Note that ``Z`` can also
represent Greenwich Mean Time (GMT) but is an acceptable representation
of UTC time. If the data requires a different time zone, this can be
accommodated but it is recommended that UTC be used whenever possible to
avoid confusion of local time and local daylight savings. Milliseconds
can be accurate to 9 decimal places. ISO dates are formatted
``YYYY-MM-DD``. Hours are given as a 24 hour number or military time,
e.g. 4:00 PM is 16:00.

Location
~~~~~~~~

All latitude and longitude locations are given in decimal degrees in the
well known datum specified at the ``Survey`` level. **NOTE: The entire
survey should use only one datum that is specified at the Survey
level.**

-  All latitude values must be :math:`<|90|` and all longitude values
   must be :math:`<|180|`.

-  Elevation and other distance values are given in meters.

-  Datum should be one of the well known datums, WGS84 is preferred, but
   others are acceptable.

Angles
~~~~~~

All angles of orientation are given in decimal degrees. Orientation of
channels should be given in a geographic or a geomagnetic reference
frame where the right-hand coordinates are assumed to be North = 0, East
= 90, and vertical is positive downward (Figure `2 <#fig:reference>`__).
The coordinate reference frame is given at the station level
``station.orientation.reference_frame``. Two angles to describe the
orientation of a sensor is given by ``channel.measurement_azimuth`` and
``channel.measurement_tilt``. In a geographic or geomagnetic reference
frame, the azimuth refers to the horizontal angle relative to north
positive clockwise, and the tilt refers to the vertical angle with
respect to the horizontal plane. In this reference frame, a tilt angle
of 90 points downward, 0 is parallel with the surface, and -90 points
upwards.

| Archived data should remain in measurement coordinates. Any
  transformation of coordinates for derived products can store the
  transformation angles at the channel level in
| ``channel.transformed_azimuth`` and ``channel.transformed_tilt``, the
  transformed reference frame can then be recorded in
  ``station.orientation.transformed_reference_frame``.

.. figure:: images/reference_frame.svg

   Diagram showing a right-handed geographic coordinate system. The
   azimuth is measured positive clockwise along the horizontal axis and
   tilt is measured from the vertical axis with positive down = 0,
   positive up = 180, and horizontal = 90.

Units
-----

Acceptable units are only those from the International System of Units
(SI). Only long names in all lower case are acceptable. Table
`1 <#tab:units>`__ summarizes common acceptable units.

.. container::
   :name: tab:units

   .. table:: Acceptable Units

      ==================== ===============
      **Measurement Type** **Unit Name**
      ==================== ===============
      Angles               decimal degrees
      Distance             meter
      Electric Field       millivolt
      Latitude/Longitude   decimal degrees
      Magnetic Field       nanotesla
      Resistance           ohms
      Resistivity          ohm-meter
      Temperature          celsius
      Time                 second
      Voltage              volt
      ==================== ===============

String Formats
--------------

Each metadata keyword can have a specific string style, such as date and
time or alpha-numeric. These are described in Table `2 <#tab:values>`__.
Note that any list should be comma separated.

.. container::
   :name: tab:values

   .. table:: Acceptable String Formats

      +----------------------+----------------------+----------------------+
      | **Style**            | **Description**      | **Example**          |
      +======================+======================+======================+
      | Free Form            | An unregulated       | This is Free Form!   |
      |                      | string that can      |                      |
      |                      | contain {a-z, A-Z,   |                      |
      |                      | 0-9} and special     |                      |
      |                      | characters           |                      |
      +----------------------+----------------------+----------------------+
      | Alpha Numeric        | A string that        | WGS84 or GEOMAG-USGS |
      |                      | contains no spaces   |                      |
      |                      | and only characters  |                      |
      |                      | {a-z, A-Z, 0-9, -,   |                      |
      |                      | /, \_}               |                      |
      +----------------------+----------------------+----------------------+
      | Controlled           | Only certain names   | reference_frame =    |
      | Vocabulary           | or words are         | geographic           |
      |                      | allowed. In this     |                      |
      |                      | case, examples of    |                      |
      |                      | acceptable values    |                      |
      |                      | are provided in the  |                      |
      |                      | documentation as [   |                      |
      |                      | option01 :math:`|`   |                      |
      |                      | option02 :math:`|`   |                      |
      |                      | ... ]. The ...       |                      |
      |                      | indicates that other |                      |
      |                      | options are possible |                      |
      |                      | but have not been    |                      |
      |                      | defined in the       |                      |
      |                      | standards yet        |                      |
      +----------------------+----------------------+----------------------+
      | List                 | List of entries      | Ex, Ey, Hx, Hy, Hz,  |
      |                      | using a comma        | T                    |
      |                      | separator            |                      |
      +----------------------+----------------------+----------------------+
      | Number               | A number according   | 10.0 (float) or 10   |
      |                      | to the data type;    | (integer)            |
      |                      | number of decimal    |                      |
      |                      | places has not been  |                      |
      |                      | implemented yet      |                      |
      +----------------------+----------------------+----------------------+
      | Date                 | ISO formatted date   | 2020-02-02           |
      |                      | YYYY-MM-DD in UTC    |                      |
      +----------------------+----------------------+----------------------+
      | Date Time            | ISO formatted date   | 2020-02-02T1         |
      |                      | time                 | 2:20:45.123456+00:00 |
      |                      | YYYY-MM-             |                      |
      |                      | DDThh:mm:ss.ms+00:00 |                      |
      |                      | in UTC               |                      |
      +----------------------+----------------------+----------------------+
      | Email                | A valid email        | `person@mt.or        |
      |                      | address              | g <person@mt.org>`__ |
      +----------------------+----------------------+----------------------+
      | URL                  | A full URL that a    | https://             |
      |                      | user can view in a   | www.passcal.nmt.edu/ |
      |                      | web browser          |                      |
      +----------------------+----------------------+----------------------+


Survey
======

A survey describes an entire data set that covers a specific time span
and region. This may include multiple PIs in multiple data collection
episodes but should be confined to a specific experiment or project. The
``Survey`` metadata category describes the general parameters of the
survey.

.. container::
   :name: tab:survey

   .. table:: Attributes for Survey
       :widths: 30 50 20
   
       +----------------------------------------------+--------------------------------+----------------+
       | **Metadata Key**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **acquired_by.author**                       | Name of the person or persons  | person name    |
       |                                              | who acquired the data.  This   |                |
       | Required: True                               | can be different from the      |                |
       |                                              | project lead if a contractor   |                |
       | Units: None                                  | or different group collected   |                |
       |                                              | the data.                      |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.comments**                     | Any comments about aspects of  | Lightning      |
       |                                              | how the data were collected or | strike caused a|
       | Required: False                              | any inconsistencies in the     | time skip at 8 |
       |                                              | data.                          | am UTC.        |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **archive_id**                               | Alphanumeric name provided by  | YKN20          |
       |                                              | the archive. For IRIS this     |                |
       | Required: True                               | will be the FDSN providing a   |                |
       |                                              | code.                          |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Alpha Numeric                         |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **archive_network**                          | Network code given by          | EM             |
       |                                              | PASSCAL/IRIS/FDSN.  This will  |                |
       | Required: True                               | be a two character String that |                |
       |                                              | describes who and where the    |                |
       | Units: None                                  | network operates.              |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Alpha Numeric                         |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **citation_dataset.doi**                     | The full URL of the doi Number | \url{http://doi|
       |                                              | provided by the archive that   | .10.adfabe     |
       | Required: True                               | describes the raw data         |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: URL                                   |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **citation_journal.doi**                     | The full URL of the doi Number | \url{http://do |
       |                                              | for a journal article(s) that  | i.10.xbsfs2}   |
       | Required: False                              | uses these data.  If multiple  |                |
       |                                              | journal articles use these     |                |
       | Units: None                                  | data provide as a comma        |                |
       |                                              | separated String of urls.      |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: URL                                   |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | Any comments about the survey  | Solar activity |
       |                                              | that are important for any     | low.           |
       | Required: False                              | user to know.                  |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **country**                                  | Country or countries that the  |  Canada        |
       |                                              | survey is located in. If       |                |
       | Required: True                               | multiple input as comma        |                |
       |                                              | separated names.               |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **datum**                                    | The reference datum for all    | WGS84          |
       |                                              | geographic coordinates         |                |
       | Required: True                               | throughout the survey. It is   |                |
       |                                              | up to the user to be sure that |                |
       | Units: None                                  | all coordinates are projected  |                |
       |                                              | into this datum.  Should be a  |                |
       | Type: String                                 | well-known datum: [ WGS84 $|$  |                |
       |                                              | NAD83 $|$ OSGB36 $|$ GDA94 $|$ |                |
       | Style: Controlled Vocabulary                 | ETRS89 $|$ PZ-90.11 $|$ ... ]  |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **geographic_name**                          | Geographic names that          | Southwestern   |
       |                                              | encompass the survey.  These   | USA            |
       | Required: True                               | should be broad geographic     |                |
       |                                              | names.  Further information    |                |
       | Units: None                                  | can be found at \url{https://w |                |
       |                                              | ww.usgs.gov/core-science-      |                |
       | Type: String                                 | systems/ngp/board-on-          |                |
       |                                              | geographic-names}              |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **name**                                     | Descriptive name of the survey | MT Characteriza|
       |                                              |                                | tion of Yukon  |
       | Required: True                               |                                | Terrane        |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **northwest_corner.latitude**                | Latitude of the northwest      | 23.134         |
       |                                              | corner of the survey in the    |                |
       | Required: True                               | datum specified.               |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **northwest_corner.longitude**               | Longitude of the northwest     | 14.23          |
       |                                              | corner of the survey in the    |                |
       | Required: True                               | datum specified.               |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **project**                                  | Alphanumeric name for the      | GEOMAG         |
       |                                              | project.  This is different    |                |
       | Required: True                               | than the archive_id in that it |                |
       |                                              | describes a project as having  |                |
       | Units: None                                  | a common project lead and      |                |
       |                                              | source of funding.  There may  |                |
       | Type: String                                 | be multiple surveys within a   |                |
       |                                              | project. For example if the    |                |
       | Style: Free Form                             | project is to estimate         |                |
       |                                              | geomagnetic hazards that       |                |
       |                                              | project = GEOMAG but the       |                |
       |                                              | archive_id = YKN20.            |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **project_lead.author**                      | Name of the project lead.      | Magneto        |
       |                                              | This should be a person who is |                |
       | Required: True                               | responsible for the data.      |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **project_lead.email**                       | Email of the project lead.     | mt.guru@em.org |
       |                                              | This is in case there are any  |                |
       | Required: True                               | questions about data.          |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Email                                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **project_lead.organization**                | Organization name of the       | MT Gurus       |
       |                                              | project lead.                  |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **release_license**                          | How the data can be used. The  | CC 0           |
       |                                              | options are based on Creative  |                |
       | Required: True                               | Commons licenses.  Options: [  |                |
       |                                              | CC 0 $|$ CC BY $|$ CC BY-SA$|$ |                |
       | Units: None                                  | CC BY-ND $|$ CC BY-NC-SA $|$   |                |
       |                                              | CC BY-NC-ND]. For details      |                |
       | Type: String                                 | visit \url{https://creativecom |                |
       |                                              | mons.org/licenses/             |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **southeast_corner.latitude**                | Latitude of the southeast      | 23.134         |
       |                                              | corner of the survey in the    |                |
       | Required: True                               | datum specified.               |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **southeast_corner.longitude**               | Longitude of the southeast     | 14.23          |
       |                                              | corner of the survey in the    |                |
       | Required: True                               | datum specified.               |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **summary**                                  | Summary paragraph of the       | Long project of|
       |                                              | survey including the purpose;  | characterizing |
       | Required: True                               | difficulties; data quality;    | mineral        |
       |                                              | summary of outcomes if the     | resources in   |
       | Units: None                                  | data have been processed and   | Yukon          |
       |                                              | modeled.                       |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end_date**                     | End date of the survey in UTC. | 2020-02-01     |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start_date**                   | Start date of the survey in    | 1995-06-21     |
       |                                              | UTC.                           |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	
Example Survey XML Element
--------------------------

::

   <?xml version="1.0" ?>
   <survey>
       <acquired_by>
           <author>MT Graduate Students</author>
           <comments>Multiple over 5 years</comments>
       </acquired_by>
       <archive_id>SAM1990</archive_id>
       <archive_network>EM</archive_network>
       <citation_dataset>
           <doi>https://doi.###</doi>
       </citation_dataset>
       <citation_journal>
           <doi>https://doi.###</doi>
       </citation_journal>
       <comments>None</comments>
       <country>USA, Canada</country>
       <datum>WGS84</datum>
       <geographic_name>Yukon</geographic_name>
       <name>Imaging Gold Deposits of the Yukon Province</name>
       <northwest_corner>
           <latitude type="Float" units="decimal degrees">-130</latitude>
           <longitude type="Float" units="decimal degrees">75.9</longitude>
       </northwest_corner>
       <project>AURORA</project>
       <project_lead>
           <Email>m.tee@mt.org</Email>
           <organization>EM Ltd.</organization>
           <author>M. Tee</author>
       </project_lead>
       <release_license>CC0</release_license>
       <southeast_corner>
           <latitude type="Float" units="decimal degrees">-110.0</latitude>
           <longitude type="Float" units="decimal degrees">65.12</longitude>
       </southeast_corner>
       <summary>This survey spanned multiple years with graduate students
                collecting the data.  Lots of curious bears and moose,
                some interesting signal from the aurora.  Modeled data
                image large scale crustal features like the 
                "fingers of god" that suggest large mineral deposits.
       </summary>
       <time_period>
           <end_date>2020-01-01</end_date>
           <start_date>1995-01-01</start_date>
       </time_period>
   </survey>

Station
=======

A station encompasses a single site where data are collected. If the
location changes during a run, then a new station should be created and
subsequently a new run under the new station. If the sensors, cables,
data logger, battery, etc. are replaced during a run but the station
remains in the same location, then this can be recorded in the ``Run``
metadata but does not require a new station entry.

.. container::
   :name: tab:station

   .. table:: Attributes for Station
   
       +----------------------------------------------+--------------------------------+----------------+
       | **Metadata Key**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **acquired_by.author**                       | Name of person or group that   | person name    |
       |                                              | collected the station data and |                |
       | Required: True                               | will be the point of contact   |                |
       |                                              | if any questions arise about   |                |
       | Units: None                                  | the data.                      |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.comments**                     | Any comments about who         | Expert diggers.|
       |                                              | acquired the data.             |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **archive_id**                               | Station name that is archived  | MT201          |
       |                                              | {a-z;A-Z;0-9.  For IRIS this   |                |
       | Required: True                               | is a 5 character String.       |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Alpha Numeric                         |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **channel_layout**                           | How the dipoles and magnetic   | +              |
       |                                              | channels of the station were   |                |
       | Required: False                              | laid out.  Options: [ L $|$ +  |                |
       |                                              | $|$ ... ]                      |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **channels_recorded**                        | List of components recorded by |  T             |
       |                                              | the station. Should be a       |                |
       | Required: True                               | summary of all channels        |                |
       |                                              | recorded dropped channels will |                |
       | Units: None                                  | be recorded in Run.  \qquad    |                |
       |                                              | Options: [ Ex $|$ Ey $|$ Hx    |                |
       | Type: String                                 | $|$ Hy $|$ Hz $|$ T $|$        |                |
       |                                              | Battery $|$ ... ]              |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | Any comments on the station    | Pipeline near  |
       |                                              | that would be important for a  | by.            |
       | Required: False                              | user.                          |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_type**                                | All types of data recorded by  | BBMT           |
       |                                              | the station. If multiple types |                |
       | Required: True                               | input as a comma separated     |                |
       |                                              | list. \qquad Options: [ RMT    |                |
       | Units: None                                  | $|$ AMT $|$ BBMT $|$ LPMT $|$  |                |
       |                                              | ULPMT $|$ ... ]                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **geographic_name**                          | Closest geographic name to the |  YK"           |
       |                                              | station                        |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **id**                                       | Station name.  This can be a   | bear hallabaloo|
       |                                              | longer name than the           |                |
       | Required: True                               | archive_id name and be a more  |                |
       |                                              | explanatory name.              |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.declination.comments**            | Any comments on declination    | Different than |
       |                                              | that are important to an end   | recorded       |
       | Required: False                              | user.                          | declination    |
       |                                              |                                | from data      |
       | Units: None                                  |                                | logger.        |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.declination.model**               | Name of the geomagnetic        | WMM-2016       |
       |                                              | reference model as             |                |
       | Required: True                               | \{model_name\\{-\\{YYYY\.      |                |
       |                                              | Model options: \qquad [ EMAG2  |                |
       | Units: None                                  | $|$ EMM $|$ HDGM $|$ IGRF $|$  |                |
       |                                              | WMM ]                          |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.declination.value**               | Declination angle relative to  | 12.3           |
       |                                              | geographic north positive      |                |
       | Required: True                               | clockwise estimated from       |                |
       |                                              | location and geomagnetic       |                |
       | Units: decimal degrees                       | model.                         |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.elevation**                       | Elevation of station location  | 123.4          |
       |                                              | in datum specified at survey   |                |
       | Required: True                               | level.                         |                |
       |                                              |                                |                |
       | Units: meters                                |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.latitude**                        | Latitude of station location   | 23.134         |
       |                                              | in datum specified at survey   |                |
       | Required: True                               | level.                         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.longitude**                       | Longitude of station location  | 14.23          |
       |                                              | in datum specified at survey   |                |
       | Required: True                               | level.                         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **orientation.method**                       | Method for orienting station   | compass        |
       |                                              | channels.  Options: [ compass  |                |
       | Required: True                               | $|$ GPS $|$ theodolite $|$     |                |
       |                                              | electric_compass $|$ ... ]     |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **orientation.reference_frame**              | Reference frame for station    | geomagnetic    |
       |                                              | layout.  There are only 2      |                |
       | Required: True                               | options geographic and         |                |
       |                                              | geomagnetic.  Both assume a    |                |
       | Units: None                                  | right-handed coordinate system |                |
       |                                              | with North=0                   |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **orientation.transformed_reference_frame**  | Reference frame rotation angel | 10             |
       |                                              | relative to                    |                |
       | Required: False                              | orientation.reference_frame    |                |
       |                                              | assuming positive clockwise.   |                |
       | Units: None                                  | Should only be used if data    |                |
       |                                              | are rotated.                   |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.comments**                      | Any comments on provenance of  | From a         |
       |                                              | the data.                      | graduated      |
       | Required: False                              |                                | graduate       |
       |                                              |                                | student.       |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.creation_time**                 | Date and time the file was     | 2020-02-08 T12:|
       |                                              | created.                       | 23:40.324600   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.log**                           | A history of any changes made  | 2020-02-10     |
       |                                              | to the data.                   | T14:24:45+00:00|
       | Required: False                              |                                | updated station|
       |                                              |                                | metadata.      |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.software.author**               | Author of the software used to | programmer 01  |
       |                                              | create the data files.         |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.software.name**                 | Name of the software used to   | mtrules        |
       |                                              | create data files              |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.software.version**              | Version of the software used   | 12.01a         |
       |                                              | to create data files           |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.submitter.author**              | Name of the person submitting  | person name    |
       |                                              | the data to the archive.       |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.submitter.email**               | Email of the person submitting | mt.guru@em.org |
       |                                              | the data to the archive.       |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Email                                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.submitter.organization**        | Name of the organization that  | MT Gurus       |
       |                                              | is submitting data to the      |                |
       | Required: True                               | archive.                       |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | Start date and time of         | 2020-02-01 T09:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

Example Station JSON
--------------------

::

   {    "station": {
           "acquired_by": {
               "author": "mt",
               "comments": null},
           "archive_id": "MT012",
           "channel_layout": "L",
           "channels_recorded": "Ex, Ey, Hx, Hy",
           "comments": null,
           "data_type": "MT",
           "geographic_name": "Whitehorse, Yukon",
           "id": "Curious Bears Hallabaloo",
           "location": {
               "latitude": 10.0,
               "longitude": -112.98,
               "elevation": 1234.0,
               "declination": {
                   "value": 12.3,
                   "comments": null,
                   "model": "WMM-2016"}},
           "orientation": {
               "method": "compass",
               "reference_frame": "geomagnetic"},
           "provenance": {
               "comments": null,
               "creation_time": "1980-01-01T00:00:00+00:00",
               "log": null,
               "software": {
                   "author": "test",
                   "version": "1.0a",
                   "name": "name"},
               "submitter": {
                   "author": "name",
                   "organization": null,
                   "email": "test@here.org"}},
           "time_period": {
               "end": "1980-01-01T00:00:00+00:00",
               "start": "1982-01-01T16:45:15+00:00"}
            }
   }

Run
===

A run represents data collected at a single station with a single
sampling rate. If the dipole length or other such station parameters are
changed between runs, this would require adding a new run. If the
station is relocated then a new station should be created. If a run has
channels that drop out, the start and end period will be the minimum
time and maximum time for all channels recorded.

.. container::
   :name: tab:run

   .. table:: Attributes for Run

       +----------------------------------------------+--------------------------------+----------------+
       | **Metadata Key**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **acquired_by.author**                       | Name of the person or persons  | M.T. Nubee     |
       |                                              | who acquired the run data.     |                |
       | Required: True                               | This can be different from the |                |
       |                                              | station.acquired_by and        |                |
       | Units: None                                  | survey.acquired_by.            |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.comments**                     | Any comments about who         | Group of       |
       |                                              | acquired the data.             | undergraduates.|
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **channels_recorded_auxiliary**              | List of auxiliary channels     |  battery       |
       |                                              | recorded.                      |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: name list                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **channels_recorded_electric**               | List of electric channels      |  Ey            |
       |                                              | recorded.                      |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: name list                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **channels_recorded_magnetic**               | List of magnetic channels      |  Hz            |
       |                                              | recorded.                      |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: name list                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | Any comments on the run that   | Badger attacked|
       |                                              | would be important for a user. | Ex.            |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | Any comments on the run that   | cows chewed    |
       |                                              | would be important for a user. | cables at 9am  |
       | Required: False                              |                                | local time.    |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.firmware.author**              | Author of the firmware that    | instrument     |
       |                                              | runs the data logger.          | engineer       |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.firmware.name**                | Name of the firmware the data  | mtrules        |
       |                                              | logger runs.                   |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.firmware.version**             | Version of the firmware that   | 12.01a         |
       |                                              | runs the data logger.          |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.id**                           | Instrument ID Number can be    | mt01           |
       |                                              | serial Number or a designated  |                |
       | Required: False                              | ID.                            |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.manufacturer**                 | Name of person or company that | MT Gurus       |
       |                                              | manufactured the data logger.  |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.model**                        | Model version of the data      | falcon5        |
       |                                              | logger.                        |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.comments**        | Any comment about the power    | Used a solar   |
       |                                              | source.                        | panel and it   |
       | Required: False                              |                                | was cloudy.    |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Name                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.id**              | Battery ID or name             | battery01      |
       |                                              |                                |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: name                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.type**            | Battery type                   | pb-acid gel    |
       |                                              |                                | cell           |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: name                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.voltage.end**     | End voltage                    | 12.1           |
       |                                              |                                |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: volts                                 |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.voltage.start**   | Starting voltage               | 14.3           |
       |                                              |                                |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: volts                                 |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.timing_system.comments**       | Any comment on timing system   | GPS locked with|
       |                                              | that might be useful for the   | internal quartz|
       | Required: False                              | user.                          | clock          |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.timing_system.drift**          | Estimated drift of the timing  | 0.001          |
       |                                              | system.                        |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: seconds                               |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.timing_system.type**           | Type of timing system used in  | GPS            |
       |                                              | the data logger.               |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.timing_system.uncertainty**    | Estimated uncertainty of the   | 0.0002         |
       |                                              | timing system.                 |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: seconds                               |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.type**                         | Type of data logger            | broadband      |
       |                                              |                                | 32-bit         |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_type**                                | Type of data recorded for this | BBMT           |
       |                                              | run.  Options: [ RMT $|$ AMT   |                |
       | Required: True                               | $|$ BBMT $|$ LPMT $|$ ULPMT    |                |
       |                                              | $|$ ... ]                      |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **id**                                       | Name of the run.  Should be    | MT302b         |
       |                                              | station name followed by an    |                |
       | Required: True                               | alphabet letter for the run.   |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Alpha Numeric                         |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **metadata_by.author**                       | Person who input the metadata. | Metadata Zen   |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **metadata_by.comments**                     | Any comments about the         | Undergraduate  |
       |                                              | metadata that would be useful  | did the input. |
       | Required: False                              | for the user.                  |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.comments**                      | Any comments on provenance of  | all good       |
       |                                              | the data that would be useful  |                |
       | Required: False                              | to users.                      |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.log**                           | A history of changes made to   | 2020-02-10     |
       |                                              | the data.                      | T14:24:45      |
       | Required: False                              |                                | +00:00 updated |
       |                                              |                                | metadata       |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **sampling_rate**                            | Sampling rate for the recorded | 100            |
       |                                              | run.                           |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: samples per second                    |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | Start date and time of         | 2020-02-01 T09:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

Example Run JSON
----------------

::

   {
       "run": {
           "acquired_by.author": "Magneto",
           "acquired_by.comments": "No hands all telekinesis.",
           "channels_recorded_auxiliary": ["temperature", "battery"],
           "channels_recorded_electric": ["Ex", "Ey"],
           "channels_recorded_magnetic": ["Hx", "Hy", "Hz"],
           "comments": "Good solar activity",
           "data_logger.firmware.author": "Engineer 01",
           "data_logger.firmware.name": "MTDL",
           "data_logger.firmware.version": "12.23a",
           "data_logger.id": "DL01",
           "data_logger.manufacturer": "MT Gurus",
           "data_logger.model": "Falcon 7",
           "data_logger.power_source.comments": "Used solar panel but cloudy",
           "data_logger.power_source.id": "Battery_07",
           "data_logger.power_source.type": "Pb-acid gel cell 72 Amp-hr",
           "data_logger.power_source.voltage.end": 14.1,
           "data_logger.power_source.voltage.start": 13.7,
           "data_logger.timing_system.comments": null,
           "data_logger.timing_system.drift": 0.000001,
           "data_logger.timing_system.type": "GPS + internal clock",
           "data_logger.timing_system.uncertainty": 0.0000001,
           "data_logger.type": "Broadband 32-bit 5 channels",
           "data_type": "BBMT",
           "id": "YKN201b",
           "metadata_by.author": "Graduate Student",
           "metadata_by.comments": "Lazy",
           "provenance.comments": "Data found on old hard drive",
           "provenance.log": "2020-01-02 Updated metadata from old records",
           "sampling_rate": 256,
           "time_period.end": "1999-06-01T15:30:00+00:00",
           "time_period.start": "1999-06-5T20:45:00+00:00"
       }
   }

Electric Channel
================

Electric channel refers to a dipole measurement of the electric field
for a single station for a single run.

.. container::
   :name: tab:electric

   .. table:: Attributes for Electric

       +----------------------------------------------+--------------------------------+----------------+
       | **Metadata Key**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **ac.end**                                   | Ending AC value; if more than  |  49.5          |
       |                                              | one measurement input as a     |                |
       | Required: False                              | list of Number [1 2 ...]       |                |
       |                                              |                                |                |
       | Units: volts                                 |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **ac.start**                                 | Starting AC value; if more     |  55.8          |
       |                                              | than one measurement input as  |                |
       | Required: False                              | a list of Number [1 2 ...]     |                |
       |                                              |                                |                |
       | Units: volts                                 |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **channel_number**                           | Channel number on the data     | 1              |
       |                                              | logger of the recorded         |                |
       | Required: True                               | channel.                       |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Integer                                |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | Any comments about the channel | Lightning storm|
       |                                              | that would be useful to a      | at 6pm local   |
       | Required: False                              | user.                          | time           |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **component**                                | Name of the component          | Ex             |
       |                                              | measured.  Options: \quad [ Ex |                |
       | Required: True                               | $|$ Ey $|$ ... ]               |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **contact_resistance.end**                   | Starting contact resistance;   |  1.8           |
       |                                              | if more than one measurement   |                |
       | Required: False                              | input as a list [1             |                |
       |                                              |                                |                |
       | Units: ohms                                  |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number list                           |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **contact_resistance.start**                 | Starting contact resistance;   |  1.4           |
       |                                              | if more than one measurement   |                |
       | Required: False                              | input as a list [1             |                |
       |                                              |                                |                |
       | Units: ohms                                  |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number list                           |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.author**               | Name of person or organization | graduate       |
       |                                              | who rated the data.            | student ace    |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.method**               | The method used to rate the    | standard       |
       |                                              | data.  Should be a descriptive | deviation      |
       | Required: False                              | name and not just the name of  |                |
       |                                              | a software package.  If a      |                |
       | Units: None                                  | rating is provided             |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.value**                | Rating from 1-5 where 1 is bad | 4              |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Integer                                |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.warning**                     | Any warnings about the data    | periodic       |
       |                                              | that should be noted for       | pipeline noise |
       | Required: False                              | users.                         |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **dc.end**                                   | Ending DC value; if more than  | 1.5            |
       |                                              | one measurement input as a     |                |
       | Required: False                              | list [1                        |                |
       |                                              |                                |                |
       | Units: volts                                 |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **dc.start**                                 | Starting DC value; if more     | 1.1            |
       |                                              | than one measurement input as  |                |
       | Required: False                              | a list [1                      |                |
       |                                              |                                |                |
       | Units: volts                                 |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **dipole_length**                            | Length of the dipole           | 55.25          |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: meters                                |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.applied**                           | Boolean if filter has been     |  True          |
       |                                              | applied or not. If more than   |                |
       | Required: True                               | one filter.                    |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Boolean                                |                                |                |
       |                                              |                                |                |
       | Style: List                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.comments**                          | Any comments on filters that   | low pass is not|
       |                                              | is important for users.        | calibrated     |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.name**                              | Name of filter applied or to   | lowpass_electr |
       |                                              | be applied. If more than one   | ic             |
       | Required: True                               | filter.                        |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: List                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_azimuth**                      | Azimuth angle of the channel   | 0              |
       |                                              | in the specified survey.orient |                |
       | Required: True                               | ation.reference_frame.         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_tilt**                         | Tilt angle of channel in surve | 0              |
       |                                              | y.orientation.reference_frame. |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **negative.elevation**                       | Elevation of negative          | 123.4          |
       |                                              | electrode in datum specified   |                |
       | Required: True                               | at survey level.               |                |
       |                                              |                                |                |
       | Units: meters                                |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **negative.id**                              | Negative electrode ID Number   | electrode01    |
       |                                              |                                |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **negative.latitude**                        | Latitude of negative electrode | 23.134         |
       |                                              | in datum specified at survey   |                |
       | Required: False                              | level.                         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **negative.longitude**                       | Longitude of negative          | 14.23          |
       |                                              | electrode in datum specified   |                |
       | Required: False                              | at survey level.               |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **negative.manufacturer**                    | Person or organization that    | Electro-Dudes  |
       |                                              | manufactured the electrode.    |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **negative.model**                           | Model version of the           | falcon5        |
       |                                              | electrode.                     |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **negative.type**                            | Type of electrode              | Ag-AgCl        |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **positive.elevation**                       | Elevation of the positive      | 123.4          |
       |                                              | electrode in datum specified   |                |
       | Required: False                              | at survey level.               |                |
       |                                              |                                |                |
       | Units: meters                                |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **positive.id**                              | Positive electrode ID Number   | electrode02    |
       |                                              |                                |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **positive.latitude**                        | Latitude of positive electrode | 23.134         |
       |                                              | in datum specified at survey   |                |
       | Required: False                              | level.                         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **positive.longitude**                       | Longitude of positive          | 14.23          |
       |                                              | electrode in datum specified   |                |
       | Required: False                              | at survey level.               |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **positive.manufacturer**                    | Name of group or person that   | Electro-Dudes  |
       |                                              | manufactured the electrode.    |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **positive.model**                           | Model version of the           | falcon5        |
       |                                              | electrode.                     |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **positive.type**                            | Type of electrode              | Pb-PbCl        |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **sample_rate**                              | Sample rate of the channel.    | 8              |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: samples per second                    |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC              | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | Start date and time of         | 2020-02-01T    |
       |                                              | collection in UTC.             | 09:23:45.453670|
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_azimuth**                      | Azimuth angle of channel that  | 0              |
       |                                              | has been transformed into a    |                |
       | Required: False                              | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | Units: decimal degrees                       | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_tilt**                         | Tilt angle of channel that has | 0              |
       |                                              | been transformed into a        |                |
       | Required: False                              | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | Units: decimal degrees                       | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **type**                                     | Data type for the channel.     | electric       |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **units**                                    | Units of the data              | counts         |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

Example Electric Channel JSON
-----------------------------

::

   {
    "electric": {
       "ac.end": 10.2,
       "ac.start": 12.1,
       "channel_number": 2,
       "comments": null,
       "component": "EX",
       "contact_resistance.end": 1.2,
       "contact_resistance.start": 1.1,
       "data_quality.rating.author": "mt",
       "data_quality.rating.method": "ml",
       "data_quality.rating.value": 4,
       "data_quality.warning": null,
       "dc.end": 1.0,
       "dc.start": 2.0,
       "dipole_length": 100.0,
       "filter.applied": [false],
       "filter.comments": null,
       "filter.name": [ "counts2mv", "lowpass"],
       "measurement_azimuth": 90.0,
       "measurement_tilt": 20.0,
       "negative.elevation": 100.0,
       "negative.id": "a",
       "negative.latitude": 12.12,
       "negative.longitude": -111.12,
       "negative.manufacturer": "test",
       "negative.model": "fats",
       "negative.type": "pb-pbcl",
       "positive.elevation": 101.0,
       "positive.id": "b",
       "positive.latitude": 12.123,
       "positive.longitude": -111.14,
       "positive.manufacturer": "test",
       "positive.model": "fats",
       "positive.type": "ag-agcl",
       "sample_rate": 256.0,
       "time_period.end": "1980-01-01T00:00:00+00:00",
       "time_period.start": "2020-01-01T00:00:00+00:00",
       "type": "electric",
       "units": "counts"
     }
   }

Magnetic Channel
================

A magnetic channel is a recording of one component of the magnetic field
at a single station for a single run.

.. container::
   :name: tab:magnetic

   .. table:: Attributes for Magnetic

       +----------------------------------------------+--------------------------------+----------------+
       | **Metadata Key**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **channel_number**                           | Channel Number on the data     | 1              |
       |                                              | logger.                        |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Integer                                |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | Any comments about the channel | Pc1 at 6pm     |
       |                                              | that would be useful to a      | local time.    |
       | Required: False                              | user.                          |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **component**                                | Name of the component          | Hx             |
       |                                              | measured.  Options: \quad [ Hx |                |
       | Required: True                               | $|$ Hy $|$ Hz $|$ ... ]        |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.author**               | Name of person or organization | graduate       |
       |                                              | who rated the data.            | student ace    |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.method**               | The method used to rate the    | standard       |
       |                                              | data.  Should be a descriptive | deviation      |
       | Required: False                              | name and not just the name of  |                |
       |                                              | a software package.  If a      |                |
       | Units: None                                  | rating is provided             |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.value**                | Rating from 1-5 where 1 is bad | 4              |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Integer                                |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.warning**                     | Any warnings about the data    | periodic       |
       |                                              | that should be noted for       | pipeline noise |
       | Required: False                              | users.                         |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.applied**                           | Boolean if filter has been     |  True          |
       |                                              | applied or not. If more than   |                |
       | Required: True                               | one filter                     |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Boolean                                |                                |                |
       |                                              |                                |                |
       | Style: List                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.comments**                          | Any comments on filters that   | low pass is not|
       |                                              | is important for users.        | calibrated     |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.name**                              | Name of filter applied or to   | lowpass_electr |
       |                                              | be applied. If more than one   | ic             |
       | Required: True                               | filter.                        |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: List                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **h_field_max.end**                          | Maximum magnetic field         | 34526.1        |
       |                                              | strength at end of             |                |
       | Required: False                              | measurement.                   |                |
       |                                              |                                |                |
       | Units: nanotesla                             |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **h_field_max.start**                        | Maximum magnetic field         | 34565.2        |
       |                                              | strength at beginning of       |                |
       | Required: False                              | measurement.                   |                |
       |                                              |                                |                |
       | Units: nanotesla                             |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **h_field_min.end**                          | Minimum magnetic field         | 50453.2        |
       |                                              | strength at end of             |                |
       | Required: False                              | measurement.                   |                |
       |                                              |                                |                |
       | Units: nanotesla                             |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **h_field_min.start**                        | Minimum magnetic field         | 40345.1        |
       |                                              | strength at beginning of       |                |
       | Required: False                              | measurement.                   |                |
       |                                              |                                |                |
       | Units: nt                                    |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.elevation**                       | elevation of magnetometer in   | 123.4          |
       |                                              | datum specified at survey      |                |
       | Required: False                              | level.                         |                |
       |                                              |                                |                |
       | Units: meters                                |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.latitude**                        | Latitude of magnetometer in    | 23.134         |
       |                                              | datum specified at survey      |                |
       | Required: False                              | level.                         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.longitude**                       | Longitude of magnetometer in   | 14.23          |
       |                                              | datum specified at survey      |                |
       | Required: False                              | level.                         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_azimuth**                      | Azimuth of channel in the      | 0              |
       |                                              | specified survey.orientation.r |                |
       | Required: True                               | eference_frame.                |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_tilt**                         | Tilt of channel in survey.orie | 0              |
       |                                              | ntation.reference_frame.       |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **sample_rate**                              | Sample rate of the channel.    | 8              |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: samples per second                    |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **sensor.id**                                | Sensor ID Number or serial     | mag01          |
       |                                              | Number.                        |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **sensor.manufacturer**                      | Person or organization that    | Magnets        |
       |                                              | manufactured the magnetic      |                |
       | Required: False                              | sensor.                        |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **sensor.model**                             | Model version of the magnetic  | falcon5        |
       |                                              | sensor.                        |                |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **sensor.type**                              | Type of magnetic sensor        | induction coil |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | Start date and time of         | 2020-02-01 T09:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_azimuth**                      | Azimuth angle of channel that  | 0              |
       |                                              | has been transformed into a    |                |
       | Required: False                              | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | Units: decimal degrees                       | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_tilt**                         | Tilt angle of channel that has | 0              |
       |                                              | been transformed into a        |                |
       | Required: False                              | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | Units: decimal degrees                       | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **type**                                     | Data type for the channel      | magnetic       |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **units**                                    | Units of the data.  if         | counts         |
       |                                              | archiving should always be     |                |
       | Required: True                               | counts.  Options: [ counts $|$ |                |
       |                                              | nanotesla ]                    |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

Example Magnetic Channel JSON
-----------------------------

::

   {    "magnetic": {
           "comments": null,
           "component": "Hz",
           "data_logger": {
               "channel_number": 2},
           "data_quality": {
               "warning": "periodic pipeline",
               "rating": {
                   "author": "M. Tee",
                   "method": "Machine Learning",
                   "value": 3}},
           "filter": {
               "name": ["counts2nT", "lowpass_mag"],
               "applied": [true, false],
               "comments": null},
           "h_field_max": {
               "start": 40000.,
               "end": 420000.},
           "h_field_min": {
               "start": 38000.,
               "end": 39500.},
           "location": {
               "latitude": 25.89,
               "longitude": -110.98,
               "elevation": 1234.5},
           "measurement_azimuth": 0.0,
           "measurement_tilt": 180.0,
           "sample_rate": 64.0,
           "sensor": {
               "id": 'spud',
               "manufacturer": "F. McAraday",
               "type": "tri-axial fluxgate",
               "model": "top hat"},
           "time_period": {
               "end": "2010-01-01T00:00:00+00:00",
               "start": "2020-01-01T00:00:00+00:00"},
           "type": "magnetic",
           "units": "nT"
       }
   }

Filters
=======

``Filters`` is a table that holds information on any filters that need
to be applied to get physical units, and/or filters that were applied to
the data to analyze the signal. This includes calibrations, notch
filters, conversion of counts to units, etc. The actual filter will be
an array of numbers contained within an array named ``name`` and
formatted according to ``type``. The preferred format for a filter is a
look-up table which programatically can be converted to other formats.

It is important to note that filters will be identified by name and must
be consistent throughout the file. Names should be descriptive and self
evident. Examples:

-  ``coil_2284`` :math:`\longrightarrow` induction coil Number 2284

-  ``counts2mv`` :math:`\longrightarrow` conversion from counts to mV

-  ``e_gain`` :math:`\longrightarrow` electric field gain

-  ``datalogger_response_024`` :math:`\longrightarrow` data logger
   Number 24 response

-  ``notch_60hz`` :math:`\longrightarrow` notch filter for 60 Hz and
   harmonics

-  ``lowpass_10hz`` :math:`\longrightarrow` low pass filter below 10 Hz

In each channel there are keys to identify filters that can or have been
applied to the data to get an appropriate signal. This can be a list of
filter names or a single filter name. An ``applied`` key also exists for
the user to input whether that filter has been applied. A single Boolean
can be provided ``True`` if all filters have been applied, or ``False``
if none of the filters have been applied. Or ``applied`` can be a list
the same length as ``names`` identifying if the filter has been applied.
For example ``name: "[counts2mv, notch_60hz, e_gain]"`` and
``applied: "[True, False, True]`` would indicate that ``counts2mv`` and
``e_gain`` have been applied but ``noth_60hz`` has not.

.. container::
   :name: tab:filter

   .. table:: Attributes for Filter

       +----------------------------------------------+--------------------------------+----------------+
       | **Metadata Key**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **type**                                     | Filter type. Options: [look up | lookup         |
       |                                              | $|$ poles zeros $|$ converter  |                |
       | Required: True                               | $|$ FIR $|$ ...]               |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **name**                                     | Unique name for the filter     | counts2mv      |
       |                                              | such that it is easy to query. |                |
       | Required: True                               | See above for some examples.   |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Alpha Numeric                         |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **units_in**                                 | The input units for the        | counts         |
       |                                              | filter. Should be SI units or  |                |
       | Required: True                               | counts.                        |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **units_out**                                | The output units for the       | millivolts     |
       |                                              | filter. Should be SI units or  |                |
       | Required: True                               | counts.                        |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **calibration_date**                         | If the filter is a calibration | 2010-01-01     |
       |                                              |                                | T00:00:00      |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Date Time                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+


Example Filter JSON
-------------------

::

   {
       "filter":{
           "type": "look up",
            "name": "counts2mv",
            "units_in": "counts",
            "units_out": "mV",
            "calibration_date": "2015-07-01",
           "comments": "Accurate to 0.001 mV"
       }
   }

Auxiliary Channels
==================

Auxiliary channels include state of health channels, temperature, etc.

.. container::
   :name: tab:auxiliary

   .. table:: Attributes for Auxiliary

       +----------------------------------------------+--------------------------------+----------------+
       | **Metadata Key**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **channel_number**                           | Channel Number on the data     | 1              |
       |                                              | logger.                        |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Integer                                |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | Any comments about the channel | Pc1 at 6pm     |
       |                                              | that would be useful to a      | local time.    |
       | Required: False                              | user.                          |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **component**                                | Name of the component          | temperature    |
       |                                              | measured.  Options: [          |                |
       | Required: True                               | temperature $|$ battery $|$    |                |
       |                                              | ... ]                          |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.author**               | Name of person or organization | graduate       |
       |                                              | who rated the data.            | student ace    |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.method**               | The method used to rate the    | standard       |
       |                                              | data.  Should be a descriptive | deviation      |
       | Required: False                              | name and not just the name of  |                |
       |                                              | a software package.  If a      |                |
       | Units: None                                  | rating is provided             |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.value**                | Rating from 1-5 where 1 is bad | 4              |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Integer                                |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.warning**                     | Any warnings about the data    | periodic       |
       |                                              | that should be noted for       | pipeline noise |
       | Required: False                              | users.                         |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.applied**                           | Boolean if filter has been     |  True          |
       |                                              | applied or not. If more than   |                |
       | Required: True                               | one filter                     |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: Boolean                                |                                |                |
       |                                              |                                |                |
       | Style: List                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.comments**                          | Any comments on filters that   | low pass is not|
       |                                              | is important for users.        | calibrated     |
       | Required: False                              |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **filter.name**                              | Name of filter applied or to   | lowpass_auxili |
       |                                              | be applied. If more than one   | ary            |
       | Required: True                               | filter                         |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: List                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.elevation**                       | Elevation of channel location  | 123.4          |
       |                                              | in datum specified at survey   |                |
       | Required: False                              | level.                         |                |
       |                                              |                                |                |
       | Units: meters                                |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.latitude**                        | Latitude of channel location   | 23.134         |
       |                                              | in datum specified at survey   |                |
       | Required: False                              | level.                         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **location.longitude**                       | Longitude of channel location  | 14.23          |
       |                                              | in datum specified at survey   |                |
       | Required: False                              | level.                         |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_azimuth**                      | Azimuth of channel in the      | 0              |
       |                                              | specified survey.orientation.r |                |
       | Required: True                               | eference_frame.                |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_tilt**                         | Tilt of channel in survey.orie | 0              |
       |                                              | ntation.reference_frame.       |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: decimal degrees                       |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **sample_rate**                              | Sample rate of the channel.    | 8              |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: samples per second                    |                                |                |
       |                                              |                                |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: time                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | Start date and time of         | 2020-02-01 T09:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | Required: True                               |                                | +00:00         |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: time                                  |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_azimuth**                      | Azimuth angle of channel that  | 0              |
       |                                              | has been transformed into a    |                |
       | Required: False                              | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | Units: decimal degrees                       | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_tilt**                         | Tilt angle of channel that has | 0              |
       |                                              | been transformed into a        |                |
       | Required: False                              | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | Units: decimal degrees                       | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | Type: Float                                  |                                |                |
       |                                              |                                |                |
       | Style: Number                                |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **type**                                     | Data type for the channel.     | temperature    |
       |                                              |                                |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Free Form                             |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
       | **units**                                    | Units of the data.  Options:   | celsius        |
       |                                              | SI units or counts.            |                |
       | Required: True                               |                                |                |
       |                                              |                                |                |
       | Units: None                                  |                                |                |
       |                                              |                                |                |
       | Type: String                                 |                                |                |
       |                                              |                                |                |
       | Style: Controlled Vocabulary                 |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

Example Auxiliary XML
---------------------

::

   <auxiliary>
       <comments>great</comments>
       <component>Temperature</component>
       <data_logger>
           <channel_number type="Integer">1</channel_number>
       </data_logger>
       <data_quality>
           <warning>None</warning>
           <rating>
               <author>mt</author>
               <method>ml</method>
               <value type="Integer">4</value>
           </rating>
       </data_quality>
       <filter>
           <name>
               <i>lowpass</i>
               <i>counts2mv</i>
           </name>
           <applied type="boolean">
               <i type="boolean">True</i>
           </applied>
           <comments>test</comments>
       </filter>
       <location>
           <latitude type="Float" units="degrees">12.324</latitude>
           <longitude type="Float" units="degrees">-112.03</longitude>
           <elevation type="Float" units="degrees">1234.0</elevation>
       </location>
       <measurement_azimuth type="Float" units="degrees">0.0</measurement_azimuth>
       <measurement_tilt type="Float" units="degrees">90.0</measurement_tilt>
       <sample_rate type="Float" units="samples per second">8.0</sample_rate>
       <time_period>
           <end>2020-01-01T00:00:00+00:00</end>
           <start>2020-01-04T00:00:00+00:00</start>
       </time_period>
       <type>auxiliary</type>
       <units>celsius</units>
   </auxiliary>

.. _appendix:

Option Definitions
==================

.. container::
   :name: em

   .. table:: Generalized electromagnetic period bands. Some overlap, use the closest definition.

      +---------------+------------------------------+---------------------------------+
      | **Data Type** | **Definition**               | **Sample Rate [samples/s]**     |
      +===============+==============================+=================================+
      | AMT           | radio magnetotellurics       | :math:`>10^{3}`                 |
      +---------------+------------------------------+---------------------------------+
      | BBMT          | broadband magnetotellurics   | :math:`10^{3}` – :math:`10^{0}` |
      +---------------+------------------------------+---------------------------------+
      | LPMT          | long-period magnetotellurics | :math:`<10^{0}`                 |
      +---------------+------------------------------+---------------------------------+


.. container::
   :name: tab:channel_types

   .. table:: These are the common channel components. More can be added.

      ================ ==========================
      **Channel Type** **Definition**
      ================ ==========================
      E                electric field measurement
      H                magnetic field measurement
      T                temperature
      Battery          battery
      SOH              state-of-health
      ================ ==========================

.. container::
   :name: tab:diretions
	
   .. table:: The convention for many MT setups follows the right-hand-rule (Figure `2 <#fig:reference>`__) with X in the northern direction, Y in the eastern direction, and Z positive down. If the setup has multiple channels in the same direction, they can be labeled with a Number. For instance, if you measure multiple electric fields Ex01, Ey01, Ex02, Ey02.

      ============= ===================
      **Direction** **Definition**
      ============= ===================
      x             north direction
      y             east direction
      z             vertical direction
      # {0–9}       variable directions
      ============= ===================


.. [1]
   **Corresponding Authors:**

   Jared Peacock (`jpeacock@usgs.gov <jpeacock@usgs.gov>`__)

   Andy Frassetto
   (`andy.frassetto@iris.edu <andy.frassetto@iris.edu>`__)
