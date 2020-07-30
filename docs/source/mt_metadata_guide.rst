====================================================
A Standard for Exchangeable Magnetotelluric Metadata
====================================================

:Author: Working Group for Data Handling and Software - PASSCAL
Magnetotelluric Program
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

.. figure:: example_mt_file_structure.pdf
   :alt: Schematic of a MT time series file structure with appropriate
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
   :name: fig:example

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

.. figure:: reference_frame.pdf
   :alt: Diagram showing a right-handed geographic coordinate system.
   The azimuth is measured positive clockwise along the horizontal axis
   and tilt is measured from the vertical axis with positive down = 0,
   positive up = 180, and horizontal = 90.
   :name: fig:reference

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

[tab:units]

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

[tab:values]

Survey
======

A survey describes an entire data set that covers a specific time span
and region. This may include multiple PIs in multiple data collection
episodes but should be confined to a specific experiment or project. The
``Survey`` metadata category describes the general parameters of the
survey.

	  +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+

**acquired_by.author**

None

String

Free Form

| & Name of the person or persons who acquired the data. This can be
  different from the project lead if a contractor or different group
  collected the data. & person name

**acquired_by.comments**

None

String

Free Form

| & Any comments about aspects of how the data were collected or any
  inconsistencies in the data. & Lightning strike caused a time skip at
  8 am UTC.

**archive_id**

None

String

Alpha Numeric

| & Alphanumeric name provided by the archive. For IRIS this will be the
  FDSN providing a code. & YKN20

**archive_network**

None

String

Alpha Numeric

| & Network code given by PASSCAL/IRIS/FDSN. This will be a two
  character String that describes who and where the network operates. &
  EM

**citation_dataset.doi**

None

String

URL

| & The full URL of the doi Number provided by the archive that
  describes the raw data & http://doi.10.adfabe

**citation_journal.doi**

None

String

URL

| & The full URL of the doi Number for a journal article(s) that uses
  these data. If multiple journal articles use these data provide as a
  comma separated String of urls. & http://doi.10.xbsfs, or
  http://doi.10.xbsfs, http://doi.10.xbsfs2

[tab:survey]

.. container::
   :name: tab:survey2

   .. table:: Attributes for Survey Continued

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **comments**         | Any comments about   | Solar activity low.  |
      |                      | the survey that are  |                      |
      | None                 | important for any    |                      |
      |                      | user to know.        |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **country**          | Country or countries | USA, Canada          |
      |                      | that the survey is   |                      |
      | None                 | located in. If       |                      |
      |                      | multiple input as    |                      |
      | String               | comma separated      |                      |
      |                      | names.               |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **datum**            | The reference datum  | WGS84                |
      |                      | for all geographic   |                      |
      | None                 | coordinates          |                      |
      |                      | throughout the       |                      |
      | String               | survey. It is up to  |                      |
      |                      | the user to be sure  |                      |
      | Controlled           | that all coordinates |                      |
      | Vocabulary           | are projected into   |                      |
      |                      | this datum. Should   |                      |
      |                      | be a well-known      |                      |
      |                      | datum: [ WGS84       |                      |
      |                      | :math:`|` NAD83      |                      |
      |                      | :math:`|` OSGB36     |                      |
      |                      | :math:`|` GDA94      |                      |
      |                      | :math:`|` ETRS89     |                      |
      |                      | :math:`|` PZ-90.11   |                      |
      |                      | :math:`|` ... ]      |                      |
      +----------------------+----------------------+----------------------+
      | **geographic_name**  | Geographic names     | Eastern Mojave,      |
      |                      | that encompass the   | Southwestern USA     |
      | None                 | survey. These should |                      |
      |                      | be broad geographic  |                      |
      | String               | names. Further       |                      |
      |                      | information can be   |                      |
      | Free Form            | found at             |                      |
      |                      | https://www          |                      |
      |                      | .usgs.gov/core-scien |                      |
      |                      | ce-systems/ngp/board |                      |
      |                      | -on-geographic-names |                      |
      +----------------------+----------------------+----------------------+
      | **name**             | Descriptive name of  | MT Characterization  |
      |                      | the survey, similar  | of Yukon Terrane     |
      | None                 | to the title of a    |                      |
      |                      | journal article.     |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **northwe            | Latitude of the      |                      |
      | st_corner.latitude** | northwest corner of  |                      |
      |                      | the survey in the    |                      |
      | decimal degrees      | datum specified.     |                      |
      |                      |                      |                      |
      | Float                |                      |                      |
      |                      |                      |                      |
      | Number               |                      |                      |
      +----------------------+----------------------+----------------------+
      | **northwes           | Longitude of the     |                      |
      | t_corner.longitude** | northwest corner of  |                      |
      |                      | the survey in the    |                      |
      | decimal degrees      | datum specified.     |                      |
      |                      |                      |                      |
      | Float                |                      |                      |
      |                      |                      |                      |
      | Number               |                      |                      |
      +----------------------+----------------------+----------------------+

[tab:survey2]

.. container::
   :name: tab:survey3

   .. table:: Attributes for Survey Continued

      +-------------------------+-------------------------+----------------+
      | **Metadata Key**        | **Description**         | **Example**    |
      +=========================+=========================+================+
      | **project**             | Alphanumeric name for   | GEOMAG         |
      |                         | the project. This is    |                |
      | None                    | different than the      |                |
      |                         | archive_id in that it   |                |
      | String                  | describes a project as  |                |
      |                         | having a common project |                |
      | Free Form               | lead and source of      |                |
      |                         | funding. There may be   |                |
      |                         | multiple surveys within |                |
      |                         | a project. For example  |                |
      |                         | if the project is to    |                |
      |                         | estimate geomagnetic    |                |
      |                         | hazards that project =  |                |
      |                         | GEOMAG but the          |                |
      |                         | archive_id = YKN20.     |                |
      +-------------------------+-------------------------+----------------+
      | **project_lead.author** | Name of the project     | Magneto        |
      |                         | lead. This should be a  |                |
      | None                    | person who is           |                |
      |                         | responsible for the     |                |
      | String                  | data.                   |                |
      |                         |                         |                |
      | Free Form               |                         |                |
      +-------------------------+-------------------------+----------------+
      | **project_lead.email**  | Email of the project    | mt.guru@em.org |
      |                         | lead. This is in case   |                |
      | None                    | there are any questions |                |
      |                         | about data.             |                |
      | String                  |                         |                |
      |                         |                         |                |
      | Email                   |                         |                |
      +-------------------------+-------------------------+----------------+
      | **proj                  | Organization name of    | MT Gurus       |
      | ect_lead.organization** | the project lead.       |                |
      |                         |                         |                |
      | None                    |                         |                |
      |                         |                         |                |
      | String                  |                         |                |
      |                         |                         |                |
      | Free Form               |                         |                |
      +-------------------------+-------------------------+----------------+
      | **release_license**     | How the data can be     | CC 0           |
      |                         | used. The options are   |                |
      | None                    | based on Creative       |                |
      |                         | Commons licenses.       |                |
      | String                  | Options: [ CC 0         |                |
      |                         | :math:`|` CC BY         |                |
      | Controlled Vocabulary   | :math:`|` CC            |                |
      |                         | BY-SA\ :math:`|` CC     |                |
      |                         | BY-ND :math:`|` CC      |                |
      |                         | BY-NC-SA :math:`|` CC   |                |
      |                         | BY-NC-ND]. For details  |                |
      |                         | visit                   |                |
      |                         | https://creati          |                |
      |                         | vecommons.org/licenses/ |                |
      +-------------------------+-------------------------+----------------+
      | **sout                  | Latitude of the         |                |
      | heast_corner.latitude** | southeast corner of the |                |
      |                         | survey in the datum     |                |
      | decimal degrees         | specified.              |                |
      |                         |                         |                |
      | Float                   |                         |                |
      |                         |                         |                |
      | Number                  |                         |                |
      +-------------------------+-------------------------+----------------+
      | **south                 | Longitude of the        |                |
      | east_corner.longitude** | southeast corner of the |                |
      |                         | survey in the datum     |                |
      | decimal degrees         | specified.              |                |
      |                         |                         |                |
      | Float                   |                         |                |
      |                         |                         |                |
      | Number                  |                         |                |
      +-------------------------+-------------------------+----------------+

[tab:survey3]

.. container::
   :name: tab:survey4

   .. table:: Attributes for Survey Continued

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **summary**          | Summary paragraph of | Long project of      |
      |                      | the survey including | characterizing       |
      | None                 | the purpose;         | mineral resources in |
      |                      | difficulties; data   | Yukon                |
      | String               | quality; summary of  |                      |
      |                      | outcomes if the data |                      |
      | Free Form            | have been processed  |                      |
      |                      | and modeled.         |                      |
      +----------------------+----------------------+----------------------+
      | **ti                 | End date of the      | -02-01               |
      | me_period.end_date** | survey in UTC.       |                      |
      |                      |                      |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Date                 |                      |                      |
      +----------------------+----------------------+----------------------+
      | **time               | Start date of the    | -06-21               |
      | _period.start_date** | survey in UTC.       |                      |
      |                      |                      |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Date                 |                      |                      |
      +----------------------+----------------------+----------------------+

[tab:survey4]

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

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **                   | Name of person or    | person name          |
      | acquired_by.author** | group that collected |                      |
      |                      | the station data and |                      |
      | None                 | will be the point of |                      |
      |                      | contact if any       |                      |
      | String               | questions arise      |                      |
      |                      | about the data.      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **ac                 | Any comments about   | Expert diggers.      |
      | quired_by.comments** | who acquired the     |                      |
      |                      | data.                |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **archive_id**       | Station name that is | MT201                |
      |                      | archived             |                      |
      | None                 | a-z;A-Z;0-9. For     |                      |
      |                      | IRIS this is a 5     |                      |
      | String               | character String.    |                      |
      |                      |                      |                      |
      | Alpha Numeric        |                      |                      |
      +----------------------+----------------------+----------------------+
      | **channel_layout**   | How the dipoles and  | +                    |
      |                      | magnetic channels of |                      |
      | None                 | the station were     |                      |
      |                      | laid out. Options: [ |                      |
      | String               | L :math:`|` +        |                      |
      |                      | :math:`|` ... ]      |                      |
      | Controlled           |                      |                      |
      | Vocabulary           |                      |                      |
      +----------------------+----------------------+----------------------+
      | *                    | List of components   | Ex, Ey, Hx, Hy, Hz,  |
      | *channels_recorded** | recorded by the      | T                    |
      |                      | station. Should be a |                      |
      | None                 | summary of all       |                      |
      |                      | channels recorded    |                      |
      | String               | dropped channels     |                      |
      |                      | will be recorded in  |                      |
      | Controlled           | Run. Options: [ Ex   |                      |
      | Vocabulary           | :math:`|` Ey         |                      |
      |                      | :math:`|` Hx         |                      |
      |                      | :math:`|` Hy         |                      |
      |                      | :math:`|` Hz         |                      |
      |                      | :math:`|` T          |                      |
      |                      | :math:`|` Battery    |                      |
      |                      | :math:`|` ... ]      |                      |
      +----------------------+----------------------+----------------------+
      | **comments**         | Any comments on the  | Pipeline near by.    |
      |                      | station that would   |                      |
      | None                 | be important for a   |                      |
      |                      | user.                |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+

[tab:station]

.. table:: Attributes for Station Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **data_type**        | All types of data    | BBMT                 |
   |                      | recorded by the      |                      |
   | None                 | station. If multiple |                      |
   |                      | types input as a     |                      |
   | String               | comma separated      |                      |
   |                      | list. Options: [ RMT |                      |
   | Controlled           | :math:`|` AMT        |                      |
   | Vocabulary           | :math:`|` BBMT       |                      |
   |                      | :math:`|` LPMT       |                      |
   |                      | :math:`|` ULPMT      |                      |
   |                      | :math:`|` ... ]      |                      |
   +----------------------+----------------------+----------------------+
   | **geographic_name**  | Closest geographic   | "Whitehorse, YK"     |
   |                      | name to the station, |                      |
   | None                 | should be rather     |                      |
   |                      | general. For further |                      |
   | String               | details about        |                      |
   |                      | geographic names see |                      |
   | Free Form            | https://www          |                      |
   |                      | .usgs.gov/core-scien |                      |
   |                      | ce-systems/ngp/board |                      |
   |                      | -on-geographic-names |                      |
   +----------------------+----------------------+----------------------+
   | **id**               | Station name. This   | bear hallabaloo      |
   |                      | can be a longer name |                      |
   | None                 | than the archive_id  |                      |
   |                      | name and be a more   |                      |
   | String               | explanatory name.    |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **location.de        | Any comments on      | Different than       |
   | clination.comments** | declination that are | recorded declination |
   |                      | important to an end  | from data logger.    |
   | None                 | user.                |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **location           | Name of the          | WMM-2016             |
   | .declination.model** | geomagnetic          |                      |
   |                      | reference model as   |                      |
   | None                 | {m                   |                      |
   |                      | odel_name}{-}{YYYY}. |                      |
   | String               | Model options:       |                      |
   |                      |                      |                      |
   | Controlled           |                      |                      |
   | Vocabulary           |                      |                      |
   +----------------------+----------------------+----------------------+
   | **location           | Declination angle    |                      |
   | .declination.value** | relative to          |                      |
   |                      | geographic north     |                      |
   | decimal degrees      | positive clockwise   |                      |
   |                      | estimated from       |                      |
   | Float                | location and         |                      |
   |                      | geomagnetic model.   |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **                   | Elevation of station |                      |
   | location.elevation** | location in datum    |                      |
   |                      | specified at survey  |                      |
   | meters               | level.               |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Station Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | *                    | Latitude of station  |                      |
   | *location.latitude** | location in datum    |                      |
   |                      | specified at survey  |                      |
   | decimal degrees      | level.               |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **                   | Longitude of station |                      |
   | location.longitude** | location in datum    |                      |
   |                      | specified at survey  |                      |
   | decimal degrees      | level.               |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **                   | Method for orienting | compass              |
   | orientation.method** | station channels.    |                      |
   |                      | Options: [ compass   |                      |
   | None                 | :math:`|` GPS        |                      |
   |                      | :math:`|` theodolite |                      |
   | String               | :math:`|`            |                      |
   |                      | electric_compass     |                      |
   | Controlled           | :math:`|` ... ]      |                      |
   | Vocabulary           |                      |                      |
   +----------------------+----------------------+----------------------+
   | **orientati          | Reference frame for  | geomagnetic          |
   | on.reference_frame** | station layout.      |                      |
   |                      | There are only 2     |                      |
   | None                 | options geographic   |                      |
   |                      | and geomagnetic.     |                      |
   | String               | Both assume a        |                      |
   |                      | right-handed         |                      |
   | Controlled           | coordinate system    |                      |
   | Vocabulary           | with North=0, E=90   |                      |
   |                      | and vertical         |                      |
   |                      | positive downward.   |                      |
   |                      | Options: [           |                      |
   |                      | geographic :math:`|` |                      |
   |                      | geomagnetic ]        |                      |
   +----------------------+----------------------+----------------------+
   | **o                  | Reference frame      |                      |
   | rientation.transform | rotation angel       |                      |
   | ed_reference_frame** | relative to          |                      |
   |                      | orienta              |                      |
   | None                 | tion.reference_frame |                      |
   |                      | assuming positive    |                      |
   | Float                | clockwise. Should    |                      |
   |                      | only be used if data |                      |
   | Number               | are rotated.         |                      |
   +----------------------+----------------------+----------------------+
   | **p                  | Any comments on      | From a graduated     |
   | rovenance.comments** | provenance of the    | graduate student.    |
   |                      | data.                |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **proven             | Date and time the    | -02-08               |
   | ance.creation_time** | file was created.    | T12:23:40.324600     |
   |                      |                      | +00:00               |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Date Time            |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Station Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **provenance.log**   | A history of any     | -02-10               |
   |                      | changes made to the  | T14:24:45+00:00      |
   | None                 | data.                | updated station      |
   |                      |                      | metadata.            |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **provenan           | Author of the        | programmer 01        |
   | ce.software.author** | software used to     |                      |
   |                      | create the data      |                      |
   | None                 | files.               |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **proven             | Name of the software | mtrules              |
   | ance.software.name** | used to create data  |                      |
   |                      | files                |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **provenanc          | Version of the       | 12.01a               |
   | e.software.version** | software used to     |                      |
   |                      | create data files    |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **provenanc          | Name of the person   | person name          |
   | e.submitter.author** | submitting the data  |                      |
   |                      | to the archive.      |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **provenan           | Email of the person  | mt.guru@em.org       |
   | ce.submitter.email** | submitting the data  |                      |
   |                      | to the archive.      |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Email                |                      |                      |
   +----------------------+----------------------+----------------------+
   | **provenance.subm    | Name of the          | MT Gurus             |
   | itter.organization** | organization that is |                      |
   |                      | submitting data to   |                      |
   | None                 | the archive.         |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Station Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **time_period.end**  | End date and time of | -02-04               |
   |                      | collection in UTC.   | T16:23:45.453670     |
   | None                 |                      | +00:00               |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Date Time            |                      |                      |
   +----------------------+----------------------+----------------------+
   | *                    | Start date and time  | -02-01               |
   | *time_period.start** | of collection in     | T09:23:45.453670     |
   |                      | UTC.                 | +00:00               |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Date Time            |                      |                      |
   +----------------------+----------------------+----------------------+

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

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **                   | Name of the person   | M.T. Nubee           |
      | acquired_by.author** | or persons who       |                      |
      |                      | acquired the run     |                      |
      | None                 | data. This can be    |                      |
      |                      | different from the   |                      |
      | String               | station.acquired_by  |                      |
      |                      | and                  |                      |
      | Free Form            | survey.acquired_by.  |                      |
      +----------------------+----------------------+----------------------+
      | **ac                 | Any comments about   | Group of             |
      | quired_by.comments** | who acquired the     | undergraduates.      |
      |                      | data.                |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **channels_          | List of auxiliary    | T, battery           |
      | recorded_auxiliary** | channels recorded.   |                      |
      |                      |                      |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | name list            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **channels           | List of electric     | Ex, Ey               |
      | _recorded_electric** | channels recorded.   |                      |
      |                      |                      |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | name list            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **channels           | List of magnetic     | Hx, Hy, Hz           |
      | _recorded_magnetic** | channels recorded.   |                      |
      |                      |                      |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | name list            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **comments**         | Any comments on the  | Badger attacked Ex.  |
      |                      | run that would be    |                      |
      | None                 | important for a      |                      |
      |                      | user.                |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+

[tab:run]

.. table:: Attributes for Run Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **comments**         | Any comments on the  | cows chewed cables   |
   |                      | run that would be    | at 9am local time.   |
   | None                 | important for a      |                      |
   |                      | user.                |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logg          | Author of the        | instrument engineer  |
   | er.firmware.author** | firmware that runs   |                      |
   |                      | the data logger.     |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_lo            | Name of the firmware | mtrules              |
   | gger.firmware.name** | the data logger      |                      |
   |                      | runs.                |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logge         | Version of the       | 12.01a               |
   | r.firmware.version** | firmware that runs   |                      |
   |                      | the data logger.     |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logger.id**   | Instrument ID Number | mt01                 |
   |                      | can be serial Number |                      |
   | None                 | or a designated ID.  |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_l             | Name of person or    | MT Gurus             |
   | ogger.manufacturer** | company that         |                      |
   |                      | manufactured the     |                      |
   | None                 | data logger.         |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | *                    | Model version of the | falcon5              |
   | *data_logger.model** | data logger.         |                      |
   |                      |                      |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Run Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **data_logger.pow    | Any comment about    | Used a solar panel   |
   | er_source.comments** | the power source.    | and it was cloudy.   |
   |                      |                      |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Name                 |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logg          | Battery ID or name   | battery01            |
   | er.power_source.id** |                      |                      |
   |                      |                      |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | name                 |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logger        | Battery type         | pb-acid gel cell     |
   | .power_source.type** |                      |                      |
   |                      |                      |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | name                 |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logger.power_ | End voltage          |                      |
   | source.voltage.end** |                      |                      |
   |                      |                      |                      |
   | volts                |                      |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **                   | Starting voltage     |                      |
   | data_logger.power_so |                      |                      |
   | urce.voltage.start** |                      |                      |
   |                      |                      |                      |
   | volts                |                      |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logger.timi   | Any comment on       | GPS locked with      |
   | ng_system.comments** | timing system that   | internal quartz      |
   |                      | might be useful for  | clock                |
   | None                 | the user.            |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logger.t      | Estimated drift of   |                      |
   | iming_system.drift** | the timing system.   |                      |
   |                      |                      |                      |
   | seconds              |                      |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Run Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **data_logger.       | Type of timing       | GPS                  |
   | timing_system.type** | system used in the   |                      |
   |                      | data logger.         |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | *                    | Estimated            |                      |
   | *data_logger.timing_ | uncertainty of the   |                      |
   | system.uncertainty** | timing system.       |                      |
   |                      |                      |                      |
   | seconds              |                      |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_logger.type** | Type of data logger, | broadband 32-bit     |
   |                      | this should specify  |                      |
   | None                 | the bit rate and any |                      |
   |                      | other parameters of  |                      |
   | String               | the data logger.     |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_type**        | Type of data         | BBMT                 |
   |                      | recorded for this    |                      |
   | None                 | run. Options: [ RMT  |                      |
   |                      | :math:`|` AMT        |                      |
   | String               | :math:`|` BBMT       |                      |
   |                      | :math:`|` LPMT       |                      |
   | Controlled           | :math:`|` ULPMT      |                      |
   | Vocabulary           | :math:`|` ... ]      |                      |
   +----------------------+----------------------+----------------------+
   | **id**               | Name of the run.     | MT302b               |
   |                      | Should be station    |                      |
   | None                 | name followed by an  |                      |
   |                      | alphabet letter for  |                      |
   | String               | the run.             |                      |
   |                      |                      |                      |
   | Alpha Numeric        |                      |                      |
   +----------------------+----------------------+----------------------+
   | **                   | Person who input the | Metadata Zen         |
   | metadata_by.author** | metadata.            |                      |
   |                      |                      |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **me                 | Any comments about   | Undergraduate did    |
   | tadata_by.comments** | the metadata that    | the input.           |
   |                      | would be useful for  |                      |
   | None                 | the user.            |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+

.. container::
   :name: tab:

   .. table:: Attributes for Run

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **p                  | Any comments on      | all good             |
      | rovenance.comments** | provenance of the    |                      |
      |                      | data that would be   |                      |
      | None                 | useful to users.     |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **provenance.log**   | A history of changes | -02-10 T14:24:45     |
      |                      | made to the data.    | +00:00 updated       |
      | None                 |                      | metadata             |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **sampling_rate**    | Sampling rate for    |                      |
      |                      | the recorded run.    |                      |
      | samples per second   |                      |                      |
      |                      |                      |                      |
      | Float                |                      |                      |
      |                      |                      |                      |
      | Number               |                      |                      |
      +----------------------+----------------------+----------------------+
      | **time_period.end**  | End date and time of | -02-04               |
      |                      | collection in UTC.   | T16:23:45.453670     |
      | None                 |                      | +00:00               |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Date Time            |                      |                      |
      +----------------------+----------------------+----------------------+
      | *                    | Start date and time  | -02-01               |
      | *time_period.start** | of collection in     | T09:23:45.453670     |
      |                      | UTC.                 | +00:00               |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Date Time            |                      |                      |
      +----------------------+----------------------+----------------------+

[tab:]

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

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **ac.end**           | Ending AC value; if  | , 49.5               |
      |                      | more than one        |                      |
      | volts                | measurement input as |                      |
      |                      | a list of Number [1  |                      |
      | Float                | 2 ...]               |                      |
      |                      |                      |                      |
      | Number               |                      |                      |
      +----------------------+----------------------+----------------------+
      | **ac.start**         | Starting AC value;   | , 55.8               |
      |                      | if more than one     |                      |
      | volts                | measurement input as |                      |
      |                      | a list of Number [1  |                      |
      | Float                | 2 ...]               |                      |
      |                      |                      |                      |
      | Number               |                      |                      |
      +----------------------+----------------------+----------------------+
      | **channel_number**   | Channel number on    |                      |
      |                      | the data logger of   |                      |
      | None                 | the recorded         |                      |
      |                      | channel.             |                      |
      | Integer              |                      |                      |
      |                      |                      |                      |
      | Number               |                      |                      |
      +----------------------+----------------------+----------------------+
      | **comments**         | Any comments about   | Lightning storm at   |
      |                      | the channel that     | 6pm local time       |
      | None                 | would be useful to a |                      |
      |                      | user.                |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **component**        | Name of the          | Ex                   |
      |                      | component measured.  |                      |
      | None                 | Options:             |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Controlled           |                      |                      |
      | Vocabulary           |                      |                      |
      +----------------------+----------------------+----------------------+
      | **cont               | Starting contact     | , 1.8                |
      | act_resistance.end** | resistance; if more  |                      |
      |                      | than one measurement |                      |
      | ohms                 | input as a list [1,  |                      |
      |                      | 2, ... ]             |                      |
      | Float                |                      |                      |
      |                      |                      |                      |
      | Number list          |                      |                      |
      +----------------------+----------------------+----------------------+

[tab:electric]

.. table:: Attributes for Electric Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **contac             | Starting contact     | , 1.4                |
   | t_resistance.start** | resistance; if more  |                      |
   |                      | than one measurement |                      |
   | ohms                 | input as a list [1,  |                      |
   |                      | 2, ... ]             |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number list          |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_qua           | Name of person or    | graduate student ace |
   | lity.rating.author** | organization who     |                      |
   |                      | rated the data.      |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **data_qua           | The method used to   | standard deviation   |
   | lity.rating.method** | rate the data.       |                      |
   |                      | Should be a          |                      |
   | None                 | descriptive name and |                      |
   |                      | not just the name of |                      |
   | String               | a software package.  |                      |
   |                      | If a rating is       |                      |
   | Free Form            | provided, the method |                      |
   |                      | should be recorded.  |                      |
   +----------------------+----------------------+----------------------+
   | **data_qu            | Rating from 1-5      |                      |
   | ality.rating.value** | where 1 is bad, 5 is |                      |
   |                      | good, and 0 is       |                      |
   | None                 | unrated. Options: [  |                      |
   |                      | 0 :math:`|` 1        |                      |
   | Integer              | :math:`|` 2          |                      |
   |                      | :math:`|` 3          |                      |
   | Number               | :math:`|` 4          |                      |
   |                      | :math:`|` 5 ]        |                      |
   +----------------------+----------------------+----------------------+
   | **da                 | Any warnings about   | periodic pipeline    |
   | ta_quality.warning** | the data that should | noise                |
   |                      | be noted for users.  |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **dc.end**           | Ending DC value; if  |                      |
   |                      | more than one        |                      |
   | volts                | measurement input as |                      |
   |                      | a list [1, 2, ... ]  |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **dc.start**         | Starting DC value;   |                      |
   |                      | if more than one     |                      |
   | volts                | measurement input as |                      |
   |                      | a list [1, 2, ... ]  |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Electric Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **dipole_length**    | Length of the dipole |                      |
   |                      |                      |                      |
   | meters               |                      |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **filter.applied**   | Boolean if filter    | True, True           |
   |                      | has been applied or  |                      |
   | None                 | not. If more than    |                      |
   |                      | one filter, input as |                      |
   | Boolean              | a comma separated    |                      |
   |                      | list. Needs to be    |                      |
   | List                 | the same length as   |                      |
   |                      | filter.name. If only |                      |
   |                      | one entry is given,  |                      |
   |                      | it is assumed to     |                      |
   |                      | apply to all filters |                      |
   |                      | listed.              |                      |
   +----------------------+----------------------+----------------------+
   | **filter.comments**  | Any comments on      | low pass is not      |
   |                      | filters that is      | calibrated           |
   | None                 | important for users. |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **filter.name**      | Name of filter       | counts2mv,           |
   |                      | applied or to be     | lowpass_electric     |
   | None                 | applied. If more     |                      |
   |                      | than one filter,     |                      |
   | String               | input as a comma     |                      |
   |                      | separated list.      |                      |
   | List                 |                      |                      |
   +----------------------+----------------------+----------------------+
   | **m                  | Azimuth angle of the |                      |
   | easurement_azimuth** | channel in the       |                      |
   |                      | specified            |                      |
   | decimal degrees      | survey.orientat      |                      |
   |                      | ion.reference_frame. |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **measurement_tilt** | Tilt angle of        |                      |
   |                      | channel in           |                      |
   | decimal degrees      | survey.orientat      |                      |
   |                      | ion.reference_frame. |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **                   | Elevation of         |                      |
   | negative.elevation** | negative electrode   |                      |
   |                      | in datum specified   |                      |
   | meters               | at survey level.     |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Electric Continued

   +-------------------------+-------------------------+---------------+
   | **Metadata Key**        | **Description**         | **Example**   |
   +=========================+=========================+===============+
   | **negative.id**         | Negative electrode ID   | electrode01   |
   |                         | Number, can be serial   |               |
   | None                    | number or a designated  |               |
   |                         | ID.                     |               |
   | String                  |                         |               |
   |                         |                         |               |
   | Free Form               |                         |               |
   +-------------------------+-------------------------+---------------+
   | **negative.latitude**   | Latitude of negative    |               |
   |                         | electrode in datum      |               |
   | decimal degrees         | specified at survey     |               |
   |                         | level.                  |               |
   | Float                   |                         |               |
   |                         |                         |               |
   | Number                  |                         |               |
   +-------------------------+-------------------------+---------------+
   | **negative.longitude**  | Longitude of negative   |               |
   |                         | electrode in datum      |               |
   | decimal degrees         | specified at survey     |               |
   |                         | level.                  |               |
   | Float                   |                         |               |
   |                         |                         |               |
   | Number                  |                         |               |
   +-------------------------+-------------------------+---------------+
   | **                      | Person or organization  | Electro-Dudes |
   | negative.manufacturer** | that manufactured the   |               |
   |                         | electrode.              |               |
   | None                    |                         |               |
   |                         |                         |               |
   | String                  |                         |               |
   |                         |                         |               |
   | Free Form               |                         |               |
   +-------------------------+-------------------------+---------------+
   | **negative.model**      | Model version of the    | falcon5       |
   |                         | electrode.              |               |
   | None                    |                         |               |
   |                         |                         |               |
   | String                  |                         |               |
   |                         |                         |               |
   | Free Form               |                         |               |
   +-------------------------+-------------------------+---------------+
   | **negative.type**       | Type of electrode,      | Ag-AgCl       |
   |                         | should specify the      |               |
   | None                    | chemistry.              |               |
   |                         |                         |               |
   | String                  |                         |               |
   |                         |                         |               |
   | Free Form               |                         |               |
   +-------------------------+-------------------------+---------------+
   | **positive.elevation**  | Elevation of the        |               |
   |                         | positive electrode in   |               |
   | meters                  | datum specified at      |               |
   |                         | survey level.           |               |
   | Float                   |                         |               |
   |                         |                         |               |
   | Number                  |                         |               |
   +-------------------------+-------------------------+---------------+

.. table:: Attributes for Electric Continued

   +-------------------------+-------------------------+---------------+
   | **Metadata Key**        | **Description**         | **Example**   |
   +=========================+=========================+===============+
   | **positive.id**         | Positive electrode ID   | electrode02   |
   |                         | Number, can be serial   |               |
   | None                    | Number or a designated  |               |
   |                         | ID.                     |               |
   | String                  |                         |               |
   |                         |                         |               |
   | Free Form               |                         |               |
   +-------------------------+-------------------------+---------------+
   | **positive.latitude**   | Latitude of positive    |               |
   |                         | electrode in datum      |               |
   | decimal degrees         | specified at survey     |               |
   |                         | level.                  |               |
   | Float                   |                         |               |
   |                         |                         |               |
   | Number                  |                         |               |
   +-------------------------+-------------------------+---------------+
   | **positive.longitude**  | Longitude of positive   |               |
   |                         | electrode in datum      |               |
   | decimal degrees         | specified at survey     |               |
   |                         | level.                  |               |
   | Float                   |                         |               |
   |                         |                         |               |
   | Number                  |                         |               |
   +-------------------------+-------------------------+---------------+
   | **                      | Name of group or person | Electro-Dudes |
   | positive.manufacturer** | that manufactured the   |               |
   |                         | electrode.              |               |
   | None                    |                         |               |
   |                         |                         |               |
   | String                  |                         |               |
   |                         |                         |               |
   | Free Form               |                         |               |
   +-------------------------+-------------------------+---------------+
   | **positive.model**      | Model version of the    | falcon5       |
   |                         | electrode.              |               |
   | None                    |                         |               |
   |                         |                         |               |
   | String                  |                         |               |
   |                         |                         |               |
   | Free Form               |                         |               |
   +-------------------------+-------------------------+---------------+
   | **positive.type**       | Type of electrode,      | Pb-PbCl       |
   |                         | should include          |               |
   | None                    | chemistry of the        |               |
   |                         | electrode.              |               |
   | String                  |                         |               |
   |                         |                         |               |
   | Free Form               |                         |               |
   +-------------------------+-------------------------+---------------+
   | **sample_rate**         | Sample rate of the      |               |
   |                         | channel.                |               |
   | samples per second      |                         |               |
   |                         |                         |               |
   | Float                   |                         |               |
   |                         |                         |               |
   | Number                  |                         |               |
   +-------------------------+-------------------------+---------------+

.. table:: Attributes for Electric Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **time_period.end**  | End date and time of | -02-04               |
   |                      | collection in UTC    | T16:23:45.453670     |
   | None                 |                      | +00:00               |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Date Time            |                      |                      |
   +----------------------+----------------------+----------------------+
   | *                    | Start date and time  | -02-01T              |
   | *time_period.start** | of collection in     | 09:23:45.453670      |
   |                      | UTC.                 | +00:00               |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Date Time            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **t                  | Azimuth angle of     |                      |
   | ransformed_azimuth** | channel that has     |                      |
   |                      | been transformed     |                      |
   | decimal degrees      | into a specified     |                      |
   |                      | coordinate system.   |                      |
   | Float                | Note this value is   |                      |
   |                      | only for derivative  |                      |
   | Number               | products from the    |                      |
   |                      | archived data.       |                      |
   +----------------------+----------------------+----------------------+
   | **transformed_tilt** | Tilt angle of        |                      |
   |                      | channel that has     |                      |
   | decimal degrees      | been transformed     |                      |
   |                      | into a specified     |                      |
   | Float                | coordinate system.   |                      |
   |                      | Note this value is   |                      |
   | Number               | only for derivative  |                      |
   |                      | products from the    |                      |
   |                      | archived data.       |                      |
   +----------------------+----------------------+----------------------+
   | **type**             | Data type for the    | electric             |
   |                      | channel.             |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **units**            | Units of the data,   | counts               |
   |                      | if archived data     |                      |
   | None                 | should always be in  |                      |
   |                      | counts. Options: [   |                      |
   | String               | counts :math:`|`     |                      |
   |                      | millivolts ]         |                      |
   | Controlled           |                      |                      |
   | Vocabulary           |                      |                      |
   +----------------------+----------------------+----------------------+

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

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **channel_number**   | Channel Number on    |                      |
      |                      | the data logger.     |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | Integer              |                      |                      |
      |                      |                      |                      |
      | Number               |                      |                      |
      +----------------------+----------------------+----------------------+
      | **comments**         | Any comments about   | Pc1 at 6pm local     |
      |                      | the channel that     | time.                |
      | None                 | would be useful to a |                      |
      |                      | user.                |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **component**        | Name of the          | Hx                   |
      |                      | component measured.  |                      |
      | None                 | Options:             |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Controlled           |                      |                      |
      | Vocabulary           |                      |                      |
      +----------------------+----------------------+----------------------+
      | **data_qua           | Name of person or    | graduate student ace |
      | lity.rating.author** | organization who     |                      |
      |                      | rated the data.      |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **data_qua           | The method used to   | standard deviation   |
      | lity.rating.method** | rate the data.       |                      |
      |                      | Should be a          |                      |
      | None                 | descriptive name and |                      |
      |                      | not just the name of |                      |
      | String               | a software package.  |                      |
      |                      | If a rating is       |                      |
      | Free Form            | provided, the method |                      |
      |                      | should be recorded.  |                      |
      +----------------------+----------------------+----------------------+
      | **data_qu            | Rating from 1-5      |                      |
      | ality.rating.value** | where 1 is bad, 5 is |                      |
      |                      | good, and 0 is       |                      |
      | None                 | unrated. Options: [  |                      |
      |                      | 0 :math:`|` 1        |                      |
      | Integer              | :math:`|` 2          |                      |
      |                      | :math:`|` 3          |                      |
      | Number               | :math:`|` 4          |                      |
      |                      | :math:`|` 5 ]        |                      |
      +----------------------+----------------------+----------------------+

[tab:magnetic]

.. table:: Attributes for Magnetic Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **da                 | Any warnings about   | periodic pipeline    |
   | ta_quality.warning** | the data that should | noise                |
   |                      | be noted for users.  |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **filter.applied**   | Boolean if filter    | True, True           |
   |                      | has been applied or  |                      |
   | None                 | not. If more than    |                      |
   |                      | one filter, input as |                      |
   | Boolean              | a comma separated    |                      |
   |                      | list. Needs to be    |                      |
   | List                 | the same length as   |                      |
   |                      | filter.name. If only |                      |
   |                      | one entry is given,  |                      |
   |                      | it is assumed to     |                      |
   |                      | apply to all filters |                      |
   |                      | listed.              |                      |
   +----------------------+----------------------+----------------------+
   | **filter.comments**  | Any comments on      | low pass is not      |
   |                      | filters that is      | calibrated           |
   | None                 | important for users. |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **filter.name**      | Name of filter       | counts2mv,           |
   |                      | applied or to be     | lowpass_electric     |
   | None                 | applied. If more     |                      |
   |                      | than one filter,     |                      |
   | String               | input as a comma     |                      |
   |                      | separated list.      |                      |
   | List                 |                      |                      |
   +----------------------+----------------------+----------------------+
   | **h_field_max.end**  | Maximum magnetic     |                      |
   |                      | field strength at    |                      |
   | nanotesla            | end of measurement.  |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | *                    | Maximum magnetic     |                      |
   | *h_field_max.start** | field strength at    |                      |
   |                      | beginning of         |                      |
   | nanotesla            | measurement.         |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **h_field_min.end**  | Minimum magnetic     |                      |
   |                      | field strength at    |                      |
   | nanotesla            | end of measurement.  |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Magnetic Continued

   +-------------------------+--------------------------+-------------+
   | **Metadata Key**        | **Description**          | **Example** |
   +=========================+==========================+=============+
   | **h_field_min.start**   | Minimum magnetic field   |             |
   |                         | strength at beginning of |             |
   | nt                      | measurement.             |             |
   |                         |                          |             |
   | Float                   |                          |             |
   |                         |                          |             |
   | Number                  |                          |             |
   +-------------------------+--------------------------+-------------+
   | **location.elevation**  | elevation of             |             |
   |                         | magnetometer in datum    |             |
   | meters                  | specified at survey      |             |
   |                         | level.                   |             |
   | Float                   |                          |             |
   |                         |                          |             |
   | Number                  |                          |             |
   +-------------------------+--------------------------+-------------+
   | **location.latitude**   | Latitude of magnetometer |             |
   |                         | in datum specified at    |             |
   | decimal degrees         | survey level.            |             |
   |                         |                          |             |
   | Float                   |                          |             |
   |                         |                          |             |
   | Number                  |                          |             |
   +-------------------------+--------------------------+-------------+
   | **location.longitude**  | Longitude of             |             |
   |                         | magnetometer in datum    |             |
   | decimal degrees         | specified at survey      |             |
   |                         | level.                   |             |
   | Float                   |                          |             |
   |                         |                          |             |
   | Number                  |                          |             |
   +-------------------------+--------------------------+-------------+
   | **measurement_azimuth** | Azimuth of channel in    |             |
   |                         | the specified            |             |
   | decimal degrees         | survey.orie              |             |
   |                         | ntation.reference_frame. |             |
   | Float                   |                          |             |
   |                         |                          |             |
   | Number                  |                          |             |
   +-------------------------+--------------------------+-------------+
   | **measurement_tilt**    | Tilt of channel in       |             |
   |                         | survey.orie              |             |
   | decimal degrees         | ntation.reference_frame. |             |
   |                         |                          |             |
   | Float                   |                          |             |
   |                         |                          |             |
   | Number                  |                          |             |
   +-------------------------+--------------------------+-------------+
   | **sample_rate**         | Sample rate of the       |             |
   |                         | channel.                 |             |
   | samples per second      |                          |             |
   |                         |                          |             |
   | Float                   |                          |             |
   |                         |                          |             |
   | Number                  |                          |             |
   +-------------------------+--------------------------+-------------+

.. table:: Attributes for Magnetic Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **sensor.id**        | Sensor ID Number or  | mag01                |
   |                      | serial Number.       |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **s                  | Person or            | Magnets              |
   | ensor.manufacturer** | organization that    |                      |
   |                      | manufactured the     |                      |
   | None                 | magnetic sensor.     |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **sensor.model**     | Model version of the | falcon5              |
   |                      | magnetic sensor.     |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **sensor.type**      | Type of magnetic     | induction coil       |
   |                      | sensor, should       |                      |
   | None                 | describe the type of |                      |
   |                      | magnetic field       |                      |
   | String               | measurement.         |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **time_period.end**  | End date and time of | -02-04               |
   |                      | collection in UTC.   | T16:23:45.453670     |
   | None                 |                      | +00:00               |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Date Time            |                      |                      |
   +----------------------+----------------------+----------------------+
   | *                    | Start date and time  | -02-01               |
   | *time_period.start** | of collection in     | T09:23:45.453670     |
   |                      | UTC.                 | +00:00               |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Date Time            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **t                  | Azimuth angle of     |                      |
   | ransformed_azimuth** | channel that has     |                      |
   |                      | been transformed     |                      |
   | decimal degrees      | into a specified     |                      |
   |                      | coordinate system.   |                      |
   | Float                | Note this value is   |                      |
   |                      | only for derivative  |                      |
   | Number               | products from the    |                      |
   |                      | archived data.       |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Magnetic Continued

   +-----------------------+--------------------------+-------------+
   | **Metadata Key**      | **Description**          | **Example** |
   +=======================+==========================+=============+
   | **transformed_tilt**  | Tilt angle of channel    |             |
   |                       | that has been            |             |
   | decimal degrees       | transformed into a       |             |
   |                       | specified coordinate     |             |
   | Float                 | system. Note this value  |             |
   |                       | is only for derivative   |             |
   | Number                | products from the        |             |
   |                       | archived data.           |             |
   +-----------------------+--------------------------+-------------+
   | **type**              | Data type for the        | magnetic    |
   |                       | channel                  |             |
   | None                  |                          |             |
   |                       |                          |             |
   | String                |                          |             |
   |                       |                          |             |
   | Free Form             |                          |             |
   +-----------------------+--------------------------+-------------+
   | **units**             | Units of the data. if    | counts      |
   |                       | archiving should always  |             |
   | None                  | be counts. Options: [    |             |
   |                       | counts :math:`|`         |             |
   | String                | nanotesla ]              |             |
   |                       |                          |             |
   | Controlled Vocabulary |                          |             |
   +-----------------------+--------------------------+-------------+

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

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **type**             | Filter type.         | lookup               |
      |                      | Options: [look up    |                      |
      | None                 | :math:`|` poles      |                      |
      |                      | zeros :math:`|`      |                      |
      | String               | converter :math:`|`  |                      |
      |                      | FIR :math:`|` ...]   |                      |
      | Controlled           |                      |                      |
      | Vocabulary           |                      |                      |
      +----------------------+----------------------+----------------------+
      | **name**             | Unique name for the  | counts2mv            |
      |                      | filter such that it  |                      |
      | None                 | is easy to query.    |                      |
      |                      | See above for some   |                      |
      | String               | examples.            |                      |
      |                      |                      |                      |
      | Alpha Numeric        |                      |                      |
      +----------------------+----------------------+----------------------+
      | **units_in**         | The input units for  | counts               |
      |                      | the filter. Should   |                      |
      | None                 | be SI units or       |                      |
      |                      | counts.              |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Controlled           |                      |                      |
      | Vocabulary           |                      |                      |
      +----------------------+----------------------+----------------------+
      | **units_out**        | The output units for | millivolts           |
      |                      | the filter. Should   |                      |
      | None                 | be SI units or       |                      |
      |                      | counts.              |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Controlled           |                      |                      |
      | Vocabulary           |                      |                      |
      +----------------------+----------------------+----------------------+
      | **calibration_date** | If the filter is a   | -01-01 T00:00:00     |
      |                      | calibration, include | +00:00               |
      | None                 | the calibration      |                      |
      |                      | date.                |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Date Time            |                      |                      |
      +----------------------+----------------------+----------------------+

[tab:filter]

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

      +----------------------+----------------------+----------------------+
      | **Metadata Key**     | **Description**      | **Example**          |
      +======================+======================+======================+
      | **channel_number**   | Channel Number on    |                      |
      |                      | the data logger.     |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | Integer              |                      |                      |
      |                      |                      |                      |
      | Number               |                      |                      |
      +----------------------+----------------------+----------------------+
      | **comments**         | Any comments about   | Pc1 at 6pm local     |
      |                      | the channel that     | time.                |
      | None                 | would be useful to a |                      |
      |                      | user.                |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **component**        | Name of the          | temperature          |
      |                      | component measured.  |                      |
      | None                 | Options: [           |                      |
      |                      | temperature          |                      |
      | String               | :math:`|` battery    |                      |
      |                      | :math:`|` ... ]      |                      |
      | Controlled           |                      |                      |
      | Vocabulary           |                      |                      |
      +----------------------+----------------------+----------------------+
      | **data_qua           | Name of person or    | graduate student ace |
      | lity.rating.author** | organization who     |                      |
      |                      | rated the data.      |                      |
      | None                 |                      |                      |
      |                      |                      |                      |
      | String               |                      |                      |
      |                      |                      |                      |
      | Free Form            |                      |                      |
      +----------------------+----------------------+----------------------+
      | **data_qua           | The method used to   | standard deviation   |
      | lity.rating.method** | rate the data.       |                      |
      |                      | Should be a          |                      |
      | None                 | descriptive name and |                      |
      |                      | not just the name of |                      |
      | String               | a software package.  |                      |
      |                      | If a rating is       |                      |
      | Free Form            | provided, the method |                      |
      |                      | should be recorded.  |                      |
      +----------------------+----------------------+----------------------+
      | **data_qu            | Rating from 1-5      |                      |
      | ality.rating.value** | where 1 is bad, 5 is |                      |
      |                      | good, and 0 is       |                      |
      | None                 | unrated. Options: [  |                      |
      |                      | 0 :math:`|` 1        |                      |
      | Integer              | :math:`|` 2          |                      |
      |                      | :math:`|` 3          |                      |
      | Number               | :math:`|` 4          |                      |
      |                      | :math:`|` 5 ]        |                      |
      +----------------------+----------------------+----------------------+

[tab:auxiliary]

.. table:: Attributes for Auxiliary Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **da                 | Any warnings about   | periodic pipeline    |
   | ta_quality.warning** | the data that should | noise                |
   |                      | be noted for users.  |                      |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **filter.applied**   | Boolean if filter    | True, True           |
   |                      | has been applied or  |                      |
   | None                 | not. If more than    |                      |
   |                      | one filter, input as |                      |
   | Boolean              | a comma separated    |                      |
   |                      | list. Needs to be    |                      |
   | List                 | the same length as   |                      |
   |                      | filter.name. If only |                      |
   |                      | one entry is given,  |                      |
   |                      | it is assumed to     |                      |
   |                      | apply to all filters |                      |
   |                      | listed.              |                      |
   +----------------------+----------------------+----------------------+
   | **filter.comments**  | Any comments on      | low pass is not      |
   |                      | filters that is      | calibrated           |
   | None                 | important for users. |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | Free Form            |                      |                      |
   +----------------------+----------------------+----------------------+
   | **filter.name**      | Name of filter       | counts2mv,           |
   |                      | applied or to be     | lowpass_auxiliary    |
   | None                 | applied. If more     |                      |
   |                      | than one filter,     |                      |
   | String               | input as a comma     |                      |
   |                      | separated list.      |                      |
   | List                 |                      |                      |
   +----------------------+----------------------+----------------------+
   | **                   | Elevation of channel |                      |
   | location.elevation** | location in datum    |                      |
   |                      | specified at survey  |                      |
   | meters               | level.               |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | *                    | Latitude of channel  |                      |
   | *location.latitude** | location in datum    |                      |
   |                      | specified at survey  |                      |
   | decimal degrees      | level.               |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **                   | Longitude of channel |                      |
   | location.longitude** | location in datum    |                      |
   |                      | specified at survey  |                      |
   | decimal degrees      | level.               |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Auxiliary Continued

   +----------------------+----------------------+----------------------+
   | **Metadata Key**     | **Description**      | **Example**          |
   +======================+======================+======================+
   | **m                  | Azimuth of channel   |                      |
   | easurement_azimuth** | in the specified     |                      |
   |                      | survey.orientat      |                      |
   | decimal degrees      | ion.reference_frame. |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **measurement_tilt** | Tilt of channel in   |                      |
   |                      | survey.orientat      |                      |
   | decimal degrees      | ion.reference_frame. |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **sample_rate**      | Sample rate of the   |                      |
   |                      | channel.             |                      |
   | samples per second   |                      |                      |
   |                      |                      |                      |
   | Float                |                      |                      |
   |                      |                      |                      |
   | Number               |                      |                      |
   +----------------------+----------------------+----------------------+
   | **time_period.end**  | End date and time of | -02-04               |
   |                      | collection in UTC.   | T16:23:45.453670     |
   | None                 |                      | +00:00               |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | time                 |                      |                      |
   +----------------------+----------------------+----------------------+
   | *                    | Start date and time  | -02-01               |
   | *time_period.start** | of collection in     | T09:23:45.453670     |
   |                      | UTC.                 | +00:00               |
   | None                 |                      |                      |
   |                      |                      |                      |
   | String               |                      |                      |
   |                      |                      |                      |
   | time                 |                      |                      |
   +----------------------+----------------------+----------------------+
   | **t                  | Azimuth angle of     |                      |
   | ransformed_azimuth** | channel that has     |                      |
   |                      | been transformed     |                      |
   | decimal degrees      | into a specified     |                      |
   |                      | coordinate system.   |                      |
   | Float                | Note this value is   |                      |
   |                      | only for derivative  |                      |
   | Number               | products from the    |                      |
   |                      | archived data.       |                      |
   +----------------------+----------------------+----------------------+
   | **transformed_tilt** | Tilt angle of        |                      |
   |                      | channel that has     |                      |
   | decimal degrees      | been transformed     |                      |
   |                      | into a specified     |                      |
   | Float                | coordinate system.   |                      |
   |                      | Note this value is   |                      |
   | Number               | only for derivative  |                      |
   |                      | products from the    |                      |
   |                      | archived data.       |                      |
   +----------------------+----------------------+----------------------+

.. table:: Attributes for Auxiliary Continued

   +-----------------------+--------------------------+-------------+
   | **Metadata Key**      | **Description**          | **Example** |
   +=======================+==========================+=============+
   | **type**              | Data type for the        | temperature |
   |                       | channel.                 |             |
   | None                  |                          |             |
   |                       |                          |             |
   | String                |                          |             |
   |                       |                          |             |
   | Free Form             |                          |             |
   +-----------------------+--------------------------+-------------+
   | **units**             | Units of the data.       | celsius     |
   |                       | Options: SI units or     |             |
   | None                  | counts.                  |             |
   |                       |                          |             |
   | String                |                          |             |
   |                       |                          |             |
   | Controlled Vocabulary |                          |             |
   +-----------------------+--------------------------+-------------+

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
   :name: tab:em

   .. table:: Generalized electromagnetic period bands. Some overlap,
   use the closest definition.

      +---------------+------------------------------+---------------------------------+
      | **Data Type** | **Definition**               | **Sample Rate [samples/s]**     |
      +===============+==============================+=================================+
      | AMT           | radio magnetotellurics       | :math:`>10^{3}`                 |
      +---------------+------------------------------+---------------------------------+
      | BBMT          | broadband magnetotellurics   | :math:`10^{3}` – :math:`10^{0}` |
      +---------------+------------------------------+---------------------------------+
      | LPMT          | long-period magnetotellurics | :math:`<10^{0}`                 |
      +---------------+------------------------------+---------------------------------+

[tab:em]

.. container::
   :name: tab:channel_types

   .. table:: These are the common channel components. More can be
   added.

      ================ ==========================
      **Channel Type** **Definition**
      ================ ==========================
      E                electric field measurement
      H                magnetic field measurement
      T                temperature
      Battery          battery
      SOH              state-of-health
      ================ ==========================

[tab:channel_types]

.. container::
   :name: tab:diretions

   .. table:: The convention for many MT setups follows the
   right-hand-rule (Figure `2 <#fig:reference>`__) with X in the
   northern direction, Y in the eastern direction, and Z positive down.
   If the setup has multiple channels in the same direction, they can be
   labeled with a Number. For instance, if you measure multiple electric
   fields Ex01, Ey01, Ex02, Ey02.

      ============= ===================
      **Direction** **Definition**
      ============= ===================
      x             north direction
      y             east direction
      z             vertical direction
      # {0–9}       variable directions
      ============= ===================

[tab:diretions]

.. [1]
   **Corresponding Authors:**

   Jared Peacock (`jpeacock@usgs.gov <jpeacock@usgs.gov>`__)

   Andy Frassetto
   (`andy.frassetto@iris.edu <andy.frassetto@iris.edu>`__)
