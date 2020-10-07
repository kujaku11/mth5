.. role:: red
.. role:: blue
.. role:: navy

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

The metadata key names should be self-explanatory and are structured
as follows:

    * ``{category}.{name}``, or can be nested
    * ``{category1}.{categroy2}.{name}`` where:
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

Archived data should remain in measurement coordinates. Any
transformation of coordinates for derived products can store the
transformation angles at the channel level in
``channel.transformed_azimuth`` and ``channel.transformed_tilt``, the
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

   .. table:: 
       :class: tight-table
       :widths: 30 45 15

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
      | List                 | List of entries      | Ex, Ey, Bx, By, Bz,  |
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

Survey Attributes
------------------



:navy:`acquired_by.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.author**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the person or persons  | person name    |
       |                                              | who acquired the data.  This   |                |
       | **Units**: None                              | can be different from the      |                |
       |                                              | project lead if a contractor   |                |
       | **Type**: String                             | or different group collected   |                |
       |                                              | the data.                      |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`acquired_by.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.comments**                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments about aspects of  | Lightning      |
       |                                              | how the data were collected or | strike caused a|
       | **Units**: None                              | any inconsistencies in the     | time skip at 8 |
       |                                              | data.                          | am UTC.        |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`fdsn.identifier`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **fdsn.identifier**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Alphanumeric name provided by  | YKN20          |
       |                                              | the archive. For IRIS this     |                |
       | **Units**: None                              | will be the FDSN providing a   |                |
       |                                              | code.                          |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Alpha Numeric                     |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`fdsn.network`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **fdsn.network**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Network code given by          | EM             |
       |                                              | PASSCAL/IRIS/FDSN.  This will  |                |
       | **Units**: None                              | be a two character String that |                |
       |                                              | describes who and where the    |                |
       | **Type**: String                             | network operates.              |                |
       |                                              |                                |                |
       | **Style**: Alpha Numeric                     |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`citation_dataset.doi`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------------+
       | **citation_dataset.doi**                     | **Description**                | **Example**          |
       +==============================================+================================+======================+
       | **Required**: :red:`True`                    | The full URL of the doi Number | http://doi.10.adfabe |
       |                                              | provided by the archive that   |                      |
       | **Units**: None                              | describes the raw data         |                      |
       |                                              |                                |                      |
       | **Type**: String                             |                                |                      |
       |                                              |                                |                      |
       | **Style**: URL                               |                                |                      |
       |                                              |                                |                      |
       |                                              |                                |                      |
       +----------------------------------------------+--------------------------------+----------------------+

:navy:`citation_journal.doi`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+-----------------------+
       | **citation_journal.doi**                     | **Description**                | **Example**           |
       +==============================================+================================+=======================+
       | **Required**: :blue:`False`                  | The full URL of the doi Number |  http://doi.10.xbsfs2 |
       |                                              | for a journal article(s) that  |                       |
       | **Units**: None                              | uses these data.  If multiple  |                       |
       |                                              | journal articles use these     |                       |
       | **Type**: String                             | data provide as a comma        |                       |
       |                                              | separated String of urls.      |                       |
       | **Style**: URL                               |                                |                       |
       |                                              |                                |                       |
       +----------------------------------------------+--------------------------------+-----------------------+

:navy:`comments`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments about the survey  | Solar activity |
       |                                              | that are important for any     | low.           |
       | **Units**: None                              | user to know.                  |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`country`
~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **country**                                  | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Country or countries that the  |  Canada        |
       |                                              | survey is located in. If       |                |
       | **Units**: None                              | multiple input as comma        |                |
       |                                              | separated names.               |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`datum`
~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **datum**                                    | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | The reference datum for all    | WGS84          |
       |                                              | geographic coordinates         |                |
       | **Units**: None                              | throughout the survey. It is   |                |
       |                                              | up to the user to be sure that |                |
       | **Type**: String                             | all coordinates are projected  |                |
       |                                              | into this datum.  Should be a  |                |
       | **Style**: Controlled Vocabulary             | well-known datum: [ WGS84  ;   |                |
       |                                              | NAD83  ;  OSGB36  ;  GDA94  ;  |                |
       |                                              | ETRS89  ;  PZ-90.11  ]         |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`geographic_name`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **geographic_name**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Geographic names that          | Southwestern,  |
       |                                              | encompass the survey.  These   | USA            |
       | **Units**: None                              | should be broad geographic     |                |
       |                                              | names.  Further information    |                |
       | **Type**: String                             | can be found at https://w      |                |
       |                                              | ww.usgs.gov/core-science-      |                |
       | **Style**: Free Form                         | systems/ngp/board-on-          |                |
       |                                              | geographic-names               |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`name`
~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **name**                                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Descriptive name of the survey | MT Characteriza|
       |                                              |                                | tion of Yukon  |
       | **Units**: None                              |                                | Terrane        |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`northwest_corner.latitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **northwest_corner.latitude**                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Latitude of the northwest      | 23.134         |
       |                                              | corner of the survey in the    |                |
       | **Units**: decimal degrees                   | datum specified.               |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`northwest_corner.longitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **northwest_corner.longitude**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Longitude of the northwest     | 14.23          |
       |                                              | corner of the survey in the    |                |
       | **Units**: decimal degrees                   | datum specified.               |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`project`
~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **project**                                  | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Alphanumeric name for the      | GEOMAG         |
       |                                              | project.  This is different    |                |
       | **Units**: None                              | than the fdsn.identifier in    |                |
       |                                              | that it describes a project    |                |
       | **Type**: String                             | with a common project lead and |                |
       |                                              | source of funding.  There may  |                |
       | **Style**: Free Form                         | be multiple surveys within a   |                |
       |                                              | project. For example if the    |                |
       |                                              | project is to estimate         |                |
       |                                              | geomagnetic hazards that       |                |
       |                                              | project = GEOMAG but the       |                |
       |                                              | fdsn.identifier = YKN20.       |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`project_lead.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **project_lead.author**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the project lead.      | Magneto        |
       |                                              | This should be a person who is |                |
       | **Units**: None                              | responsible for the data.      |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`project_lead.email`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **project_lead.email**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Email of the project lead.     | mt.guru@em.org |
       |                                              | This is in case there are any  |                |
       | **Units**: None                              | questions about data.          |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Email                             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`project_lead.organization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **project_lead.organization**                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Organization name of the       | MT Gurus       |
       |                                              | project lead.                  |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`release_license`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+---------------------------------------+----------------+
       | **release_license**                          | **Description**                       | **Example**    |
       +==============================================+=======================================+================+
       | **Required**: :red:`True`                    | How the data can be used. The         | CC-0           |
       |                                              | options are based on Creative         |                |
       | **Units**: None                              | Commons licenses.  Options -->        |                |
       |                                              | [CC-0; CC-BY; CC-BY-SA; CC-BY-ND;     |                |
       | **Type**: String                             | CC-BY-NC-SA; CC-BY-NC-ND]             |                |
       |                                              | For details visit,                    |                |
       | **Style**: Controlled Vocabulary             |                                       |                |
       |                                              | https://creativecommons.org/licenses/ |                |
       |                                              |                                       |                |
       +----------------------------------------------+---------------------------------------+----------------+

:navy:`southeast_corner.latitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **southeast_corner.latitude**                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Latitude of the southeast      | 23.134         |
       |                                              | corner of the survey in the    |                |
       | **Units**: decimal degrees                   | datum specified.               |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`southeast_corner.longitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **southeast_corner.longitude**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Longitude of the southeast     | 14.23          |
       |                                              | corner of the survey in the    |                |
       | **Units**: decimal degrees                   | datum specified.               |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`summary`
~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **summary**                                  | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Summary paragraph of the       | Long project of|
       |                                              | survey including the purpose;  | characterizing |
       | **Units**: None                              | difficulties; data quality;    | mineral        |
       |                                              | summary of outcomes if the     | resources in   |
       | **Type**: String                             | data have been processed and   | Yukon          |
       |                                              | modeled.                       |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.end_date`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end_date**                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | End date of the survey in UTC. | 2020-02-01     |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.start_date`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start_date**                   | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Start date of the survey in    | 1995-06-21     |
       |                                              | UTC.                           |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
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
       <fdsn>
           <identifier>SAM1990</identifier>
           <network>EM</network>
       </fdsn> 
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

Station Attributes
-------------------

:navy:`acquired_by.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.author**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of person or group that   | person name    |
       |                                              | collected the station data and |                |
       | **Units**: None                              | will be the point of contact   |                |
       |                                              | if any questions arise about   |                |
       | **Type**: String                             | the data.                      |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`acquired_by.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.comments**                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments about who         | Expert diggers.|
       |                                              | acquired the data.             |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+


:navy:`channel_layout`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **channel_layout**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | How the dipoles and magnetic   | "+"            |
       |                                              | channels of the station were   |                |
       | **Units**: None                              | laid out.  Options: ["L"; "+"] |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`channels_recorded`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **channels_recorded**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | List of components recorded by |  T             |
       |                                              | the station. Should be a       |                |
       | **Units**: None                              | summary of all channels        |                |
       |                                              | recorded dropped channels will |                |
       | **Type**: String                             | be recorded in Run.            |                |
       |                                              | Options:                       |                |
       | **Style**: Controlled Vocabulary             | [ Ex;  Ey; Hx; Hy; Hz; T       |                |
       |                                              | Battery; other  ]              |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`comments`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments on the station    | Pipeline near  |
       |                                              | that would be important for a  | by.            |
       | **Units**: None                              | user.                          |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_type`
~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_type**                                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | All types of data recorded by  | BBMT           |
       |                                              | the station. If multiple types |                |
       | **Units**: None                              | input as a comma separated     |                |
       |                                              | list. Options -->              |                |
       | **Type**: String                             | [RMT; AMT; BBMT; LPMT]         |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	   
:navy:`fdsn.identifier`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **fdsn.identifier**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Station name that is archived  | MT201          |
       |                                              | {a-z;A-Z;0-9.  For IRIS this   |                |
       | **Units**: None                              | is a 5 character String.       |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Alpha Numeric                     |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`geographic_name`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **geographic_name**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Closest geographic name to the |  YK"           |
       |                                              | station                        |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`id`
~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **id**                                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Station name.  This can be a   | bear hallabaloo|
       |                                              | longer name than the           |                |
       | **Units**: None                              | fdsn.identifier name and be a  |                |
       |                                              | more explanatory name.         |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.declination.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.declination.comments**            | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments on declination    | Different than |
       |                                              | that are important to an end   | recorded       |
       | **Units**: None                              | user.                          | declination    |
       |                                              |                                | from data      |
       | **Type**: String                             |                                | logger.        |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.declination.model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.declination.model**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the geomagnetic        | WMM-2016       |
       |                                              | reference model as             |                |
       | **Units**: None                              | model_name-YYYY.               |                |
       |                                              | Model options ->               |                |
       | **Type**: String                             | [EMAG2; EMM; HDGM; IGRF; WMM]  |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.declination.value`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.declination.value**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Declination angle relative to  | 12.3           |
       |                                              | geographic north positive      |                |
       | **Units**: decimal degrees                   | clockwise estimated from       |                |
       |                                              | location and geomagnetic       |                |
       | **Type**: Float                              | model.                         |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.elevation`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.elevation**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Elevation of station location  | 123.4          |
       |                                              | in datum specified at survey   |                |
       | **Units**: meters                            | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.latitude`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.latitude**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Latitude of station location   | 23.134         |
       |                                              | in datum specified at survey   |                |
       | **Units**: decimal degrees                   | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.longitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.longitude**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Longitude of station location  | 14.23          |
       |                                              | in datum specified at survey   |                |
       | **Units**: decimal degrees                   | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`orientation.method`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **orientation.method**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Method for orienting station   | compass        |
       |                                              | channels.  Options:            |                |
       | **Units**: None                              | [compass; GPS; theodolite;     |                |
       |                                              | electric_compass ]             |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`orientation.reference_frame`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **orientation.reference_frame**              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Reference frame for station    | geomagnetic    |
       |                                              | layout.  There are only 2      |                |
       | **Units**: None                              | options geographic and         |                |
       |                                              | geomagnetic.  Both assume a    |                |
       | **Type**: String                             | right-handed coordinate system |                |
       |                                              | with North=0                   |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`orientation.transformed_reference_frame`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **orientation.transformed_reference_frame**  | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Reference frame rotation angel | 10             |
       |                                              | relative to                    |                |
       | **Units**: None                              | orientation.reference_frame    |                |
       |                                              | assuming positive clockwise.   |                |
       | **Type**: Float                              | Should only be used if data    |                |
       |                                              | are rotated.                   |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.comments**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments on provenance of  | From a         |
       |                                              | the data.                      | graduated      |
       | **Units**: None                              |                                | graduate       |
       |                                              |                                | student.       |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.creation_time`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.creation_time**                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Date and time the file was     | 2020-02-08 T12:|
       |                                              | created.                       | 23:40.324600   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.log`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.log**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | A history of any changes made  | 2020-02-10     |
       |                                              | to the data.                   | T14:24:45+00:00|
       | **Units**: None                              |                                | updated station|
       |                                              |                                | metadata.      |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.software.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.software.author**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Author of the software used to | programmer 01  |
       |                                              | create the data files.         |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.software.name`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.software.name**                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the software used to   | mtrules        |
       |                                              | create data files              |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.software.version`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.software.version**              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Version of the software used   | 12.01a         |
       |                                              | to create data files           |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.submitter.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.submitter.author**              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the person submitting  | person name    |
       |                                              | the data to the archive.       |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.submitter.email`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.submitter.email**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Email of the person submitting | mt.guru@em.org |
       |                                              | the data to the archive.       |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Email                             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.submitter.organization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.submitter.organization**        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the organization that  | MT Gurus       |
       |                                              | is submitting data to the      |                |
       | **Units**: None                              | archive.                       |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.end`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.start`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Start date and time of         | 2020-02-01 T09:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+


Example Station JSON
--------------------

::

   {    "station": {
           "acquired_by": {
               "author": "mt",
               "comments": null},
           "channel_layout": "L",
           "channels_recorded": "Ex, Ey, Bx, By",
           "comments": null,
           "data_type": "MT",
           "fdsn.identifier": "MT012",
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

Run Attributes
---------------

:navy:`acquired_by.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.author**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the person or persons  | M.T. Nubee     |
       |                                              | who acquired the run data.     |                |
       | **Units**: None                              | This can be different from the |                |
       |                                              | station.acquired_by and        |                |
       | **Type**: String                             | survey.acquired_by.            |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`acquired_by.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **acquired_by.comments**                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments about who         | Group of       |
       |                                              | acquired the data.             | undergraduates.|
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`channels_recorded_auxiliary`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **channels_recorded_auxiliary**              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | List of auxiliary channels     |  battery       |
       |                                              | recorded.                      |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: name list                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`channels_recorded_electric`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **channels_recorded_electric**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | List of electric channels      |  Ey            |
       |                                              | recorded.                      |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: name list                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`channels_recorded_magnetic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **channels_recorded_magnetic**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | List of magnetic channels      |  Hz            |
       |                                              | recorded.                      |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: name list                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`comments`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments on the run that   | Badger attacked|
       |                                              | would be important for a user. | Ex.            |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+


:navy:`data_logger.firmware.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.firmware.author**              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Author of the firmware that    | instrument     |
       |                                              | runs the data logger.          | engineer       |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.firmware.name`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.firmware.name**                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Name of the firmware the data  | mtrules        |
       |                                              | logger runs.                   |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.firmware.version`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.firmware.version**             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Version of the firmware that   | 12.01a         |
       |                                              | runs the data logger.          |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.id`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.id**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Instrument ID Number can be    | mt01           |
       |                                              | serial Number or a designated  |                |
       | **Units**: None                              | ID.                            |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.manufacturer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.manufacturer**                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of person or company that | MT Gurus       |
       |                                              | manufactured the data logger.  |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.model`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.model**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Model version of the data      | falcon5        |
       |                                              | logger.                        |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.power_source.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.comments**        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comment about the power    | Used a solar   |
       |                                              | source.                        | panel and it   |
       | **Units**: None                              |                                | was cloudy.    |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Name                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.power_source.id`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.id**              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Battery ID or name             | battery01      |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: name                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.power_source.type`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.type**            | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Battery type                   | pb-acid gel    |
       |                                              |                                | cell           |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: name                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.power_source.voltage.end`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.voltage.end**     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | End voltage                    | 12.1           |
       |                                              |                                |                |
       | **Units**: volts                             |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.power_source.voltage.start`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.power_source.voltage.start**   | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Starting voltage               | 14.3           |
       |                                              |                                |                |
       | **Units**: volts                             |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.timing_system.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.timing_system.comments**       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comment on timing system   | GPS locked with|
       |                                              | that might be useful for the   | internal quartz|
       | **Units**: None                              | user.                          | clock          |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.timing_system.drift`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.timing_system.drift**          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Estimated drift of the timing  | 0.001          |
       |                                              | system.                        |                |
       | **Units**: seconds                           |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.timing_system.type`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.timing_system.type**           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Type of timing system used in  | GPS            |
       |                                              | the data logger.               |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.timing_system.uncertainty`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.timing_system.uncertainty**    | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Estimated uncertainty of the   | 0.0002         |
       |                                              | timing system.                 |                |
       | **Units**: seconds                           |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_logger.type`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_logger.type**                         | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Type of data logger            | broadband      |
       |                                              |                                | 32-bit         |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_type`
~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_type**                                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Type of data recorded for this | BBMT           |
       |                                              | run.  Options ->               |                |
       | **Units**: None                              | [RMT; AMT; BBMT; LPMT]         |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`fdsn.new_epoch`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_type**                                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Boolean if a new epoch should  | False          |
       |                                              | be made.  An epoch is a run    |                |
       | **Units**: None                              | and if parameters of the run   |                |
       |                                              | changes  set to True           |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+	   


:navy:`id`
~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **id**                                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the run.  Should be    | MT302b         |
       |                                              | station name followed by an    |                |
       | **Units**: None                              | alphabet letter for the run.   |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Alpha Numeric                     |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`metadata_by.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **metadata_by.author**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Person who input the metadata. | Metadata Zen   |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`metadata_by.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **metadata_by.comments**                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments about the         | Undergraduate  |
       |                                              | metadata that would be useful  | did the input. |
       | **Units**: None                              | for the user.                  |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.comments`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.comments**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments on provenance of  | all good       |
       |                                              | the data that would be useful  |                |
       | **Units**: None                              | to users.                      |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`provenance.log`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **provenance.log**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | A history of changes made to   | 2020-02-10     |
       |                                              | the data.                      | T14:24:45      |
       | **Units**: None                              |                                | +00:00 updated |
       |                                              |                                | metadata       |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`sampling_rate`
~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **sampling_rate**                            | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Sampling rate for the recorded | 100            |
       |                                              | run.                           |                |
       | **Units**: samples per second                |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.end`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.start`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Start date and time of         | 2020-02-01 T09:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
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
           "channels_recorded_magnetic": ["Bx", "By", "Bz"],
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
           "fdsn.new_epoch": "False",
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

Electric Channel Attributes
----------------------------

:navy:`ac.end`
~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **ac.end**                                   | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Ending AC value; if more than  |  49.5          |
       |                                              | one measurement input as a     |                |
       | **Units**: volts                             | list of Number [1 2 ...]       |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`ac.start`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **ac.start**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Starting AC value; if more     |  55.8          |
       |                                              | than one measurement input as  |                |
       | **Units**: volts                             | a list of Number [1 2 ...]     |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`channel_number`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **channel_number**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Channel number on the data     | 1              |
       |                                              | logger of the recorded         |                |
       | **Units**: None                              | channel.                       |                |
       |                                              |                                |                |
       | **Type**: Integer                            |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`comments`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments about the channel | Lightning storm|
       |                                              | that would be useful to a      | at 6pm local   |
       | **Units**: None                              | user.                          | time           |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`component`
~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **component**                                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the component          | Ex             |
       |                                              | measured.  Options: [ Ex; Ey ] |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`contact_resistance.end`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **contact_resistance.end**                   | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Starting contact resistance;   |  1.8           |
       |                                              | if more than one measurement   |                |
       | **Units**: ohms                              | input as a list [1             |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number list                       |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`contact_resistance.start`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **contact_resistance.start**                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Starting contact resistance;   |  1.4           |
       |                                              | if more than one measurement   |                |
       | **Units**: ohms                              | input as a list [1             |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number list                       |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.author**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Name of person or organization | graduate       |
       |                                              | who rated the data.            | student ace    |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.method`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.method**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | The method used to rate the    | standard       |
       |                                              | data.  Should be a descriptive | deviation      |
       | **Units**: None                              | name and not just the name of  |                |
       |                                              | a software package.  If a      |                |
       | **Type**: String                             | rating is provided             |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.value`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.value**                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Rating from 1-5 where 1 is bad | 4              |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: Integer                            |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.warning`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.warning**                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any warnings about the data    | periodic       |
       |                                              | that should be noted for       | pipeline noise |
       | **Units**: None                              | users.                         |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`dc.end`
~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **dc.end**                                   | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Ending DC value; if more than  | 1.5            |
       |                                              | one measurement input as a     |                |
       | **Units**: volts                             | list [1                        |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`dc.start`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **dc.start**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Starting DC value; if more     | 1.1            |
       |                                              | than one measurement input as  |                |
       | **Units**: volts                             | a list [1                      |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`dipole_length`
~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **dipole_length**                            | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Length of the dipole           | 55.25          |
       |                                              |                                |                |
       | **Units**: meters                            |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	   
:navy:`fdsn.channel_code`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **fdsn.channel_code**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | FDSN channel code, this is a   | LQN            |
       |                                              | 3 character code in the form   |                |
       | **Units**: None                              | [band][type][direction]        |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Alpha-Numeric                     |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	   
.. seealso:: https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/ for more information on channel codes.

:navy:`filter.applied`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.applied**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Boolean if filter has been     |  [True, False] |
       |                                              | applied or not. If more than   |                |
       | **Units**: None                              | one filter input as a list     |                |
       |                                              | that matches filter.names      |                |
       | **Type**: Boolean                            |                                |                |
       |                                              |                                |                |
       | **Style**: List                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`filter.comments`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.comments**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments on filters that   | low pass is not|
       |                                              | is important for users.        | calibrated     |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`filter.name`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.name**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of filter applied or to   | [gain,         |
       |                                              | be applied. If more than one   |  highpass_e]   |
       | **Units**: None                              | filter input as a list in the  |                |
       |                                              | order in which the should be   |                |
       | **Type**: String                             | applied.                       |                |
       |                                              |                                |                |
       | **Style**: List                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`measurement_azimuth`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_azimuth**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Azimuth angle of the channel   | 0              |
       |                                              | in the specified survey.orient |                |
       | **Units**: decimal degrees                   | ation.reference_frame.         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`measurement_tilt`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_tilt**                         | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Tilt angle of channel in surve | 0              |
       |                                              | y.orientation.reference_frame. |                |
       | **Units**: decimal degrees                   |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`negative.elevation`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **negative.elevation**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Elevation of negative          | 123.4          |
       |                                              | electrode in datum specified   |                |
       | **Units**: meters                            | at survey level.               |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`negative.id`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **negative.id**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Negative electrode ID Number   | electrode01    |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`negative.latitude`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **negative.latitude**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Latitude of negative electrode | 23.134         |
       |                                              | in datum specified at survey   |                |
       | **Units**: decimal degrees                   | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`negative.longitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **negative.longitude**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Longitude of negative          | 14.23          |
       |                                              | electrode in datum specified   |                |
       | **Units**: decimal degrees                   | at survey level.               |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`negative.manufacturer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **negative.manufacturer**                    | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Person or organization that    | Electro-Dudes  |
       |                                              | manufactured the electrode.    |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`negative.model`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **negative.model**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Model version of the           | falcon5        |
       |                                              | electrode.                     |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`negative.type`
~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **negative.type**                            | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Type of electrode              | Ag-AgCl        |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`positive.elevation`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **positive.elevation**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Elevation of the positive      | 123.4          |
       |                                              | electrode in datum specified   |                |
       | **Units**: meters                            | at survey level.               |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`positive.id`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **positive.id**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Positive electrode ID Number   | electrode02    |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`positive.latitude`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **positive.latitude**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Latitude of positive electrode | 23.134         |
       |                                              | in datum specified at survey   |                |
       | **Units**: decimal degrees                   | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`positive.longitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **positive.longitude**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Longitude of positive          | 14.23          |
       |                                              | electrode in datum specified   |                |
       | **Units**: decimal degrees                   | at survey level.               |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`positive.manufacturer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **positive.manufacturer**                    | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Name of group or person that   | Electro-Dudes  |
       |                                              | manufactured the electrode.    |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`positive.model`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **positive.model**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Model version of the           | falcon5        |
       |                                              | electrode.                     |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`positive.type`
~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **positive.type**                            | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Type of electrode              | Pb-PbCl        |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`sample_rate`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **sample_rate**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Sample rate of the channel.    | 8              |
       |                                              |                                |                |
       | **Units**: samples per second                |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.end`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC              | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.start`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Start date and time of         | 2020-02-01T    |
       |                                              | collection in UTC.             | 09:23:45.453670|
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`transformed_azimuth`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_azimuth**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Azimuth angle of channel that  | 0              |
       |                                              | has been transformed into a    |                |
       | **Units**: decimal degrees                   | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | **Type**: Float                              | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`transformed_tilt`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_tilt**                         | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Tilt angle of channel that has | 0              |
       |                                              | been transformed into a        |                |
       | **Units**: decimal degrees                   | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | **Type**: Float                              | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`type`
~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **type**                                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Data type for the channel.     | electric       |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`units`
~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **units**                                    | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Units of the data              | counts         |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
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
	   "fdsn.channel_code": "LQN",
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

Magnetic Channel Attributes
----------------------------

:navy:`channel_number`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **channel_number**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Channel Number on the data     | 1              |
       |                                              | logger.                        |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: Integer                            |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`comments`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments about the channel | Pc1 at 6pm     |
       |                                              | that would be useful to a      | local time.    |
       | **Units**: None                              | user.                          |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`component`
~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **component**                                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the component          | Hx             |
       |                                              | measured.  Options ->          |                |
       | **Units**: None                              | [Hx; Hy; Hz]                   |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.author**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Name of person or organization | graduate       |
       |                                              | who rated the data.            | student ace    |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.method`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.method**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | The method used to rate the    | standard       |
       |                                              | data.  Should be a descriptive | deviation      |
       | **Units**: None                              | name and not just the name of  |                |
       |                                              | a software package.  If a      |                |
       | **Type**: String                             | rating is provided             |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.value`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.value**                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Rating from 1-5 where 1 is bad | 4              |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: Integer                            |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.warning`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.warning**                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any warnings about the data    | periodic       |
       |                                              | that should be noted for       | pipeline noise |
       | **Units**: None                              | users.                         |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	   
:navy:`fdsn.channel_code`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **fdsn.channel_code**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | FDSN channel code, this is a   | LQN            |
       |                                              | 3 character code in the form   |                |
       | **Units**: None                              | [band][type][direction]        |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Alpha-Numeric                     |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	   
.. seealso:: https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/ for more information on channel codes.
	   

:navy:`filter.applied`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.applied**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Boolean if filter has been     |  [True, False] |
       |                                              | applied or not. If more than   |                |
       | **Units**: None                              | one filter input as a list     |                |
       |                                              | that matches filter.names      |                |
       | **Type**: Boolean                            |                                |                |
       |                                              |                                |                |
       | **Style**: List                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`filter.comments`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.comments**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments on filters that   | low pass is not|
       |                                              | is important for users.        | calibrated     |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`filter.name`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.name**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of filter applied or to   | [gain,         |
       |                                              | be applied. If more than one   | lowpass_h]     |
       | **Units**: None                              | filter input as a list in the  |                |
       |                                              | order in which the should be   |                |
       | **Type**: String                             | applied.                       |                |
       |                                              |                                |                |
       | **Style**: List                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	   
:navy:`h_field_max.end`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **h_field_max.end**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Maximum magnetic field         | 34526.1        |
       |                                              | strength at end of             |                |
       | **Units**: nanotesla                         | measurement.                   |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`h_field_max.start`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **h_field_max.start**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Maximum magnetic field         | 34565.2        |
       |                                              | strength at beginning of       |                |
       | **Units**: nanotesla                         | measurement.                   |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`h_field_min.end`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **h_field_min.end**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Minimum magnetic field         | 50453.2        |
       |                                              | strength at end of             |                |
       | **Units**: nanotesla                         | measurement.                   |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`h_field_min.start`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **h_field_min.start**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Minimum magnetic field         | 40345.1        |
       |                                              | strength at beginning of       |                |
       | **Units**: nt                                | measurement.                   |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.elevation`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.elevation**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | elevation of magnetometer in   | 123.4          |
       |                                              | datum specified at survey      |                |
       | **Units**: meters                            | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.latitude`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.latitude**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Latitude of magnetometer in    | 23.134         |
       |                                              | datum specified at survey      |                |
       | **Units**: decimal degrees                   | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.longitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.longitude**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Longitude of magnetometer in   | 14.23          |
       |                                              | datum specified at survey      |                |
       | **Units**: decimal degrees                   | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`measurement_azimuth`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_azimuth**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Azimuth of channel in the      | 0              |
       |                                              | specified survey.orientation.r |                |
       | **Units**: decimal degrees                   | eference_frame.                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`measurement_tilt`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_tilt**                         | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Tilt of channel in survey.orie | 0              |
       |                                              | ntation.reference_frame.       |                |
       | **Units**: decimal degrees                   |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`sample_rate`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **sample_rate**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Sample rate of the channel.    | 8              |
       |                                              |                                |                |
       | **Units**: samples per second                |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`sensor.id`
~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **sensor.id**                                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Sensor ID Number or serial     | mag01          |
       |                                              | Number.                        |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`sensor.manufacturer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **sensor.manufacturer**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Person or organization that    | Magnets        |
       |                                              | manufactured the magnetic      |                |
       | **Units**: None                              | sensor.                        |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`sensor.model`
~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **sensor.model**                             | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Model version of the magnetic  | falcon5        |
       |                                              | sensor.                        |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`sensor.type`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **sensor.type**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Type of magnetic sensor        | induction coil |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.end`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.start`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Start date and time of         | 2020-02-01 T09:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`transformed_azimuth`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_azimuth**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Azimuth angle of channel that  | 0              |
       |                                              | has been transformed into a    |                |
       | **Units**: decimal degrees                   | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | **Type**: Float                              | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`transformed_tilt`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_tilt**                         | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Tilt angle of channel that has | 0              |
       |                                              | been transformed into a        |                |
       | **Units**: decimal degrees                   | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | **Type**: Float                              | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`type`
~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **type**                                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Data type for the channel      | magnetic       |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`units`
~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **units**                                    | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Units of the data.  if         | counts         |
       |                                              | archiving should always be     |                |
       | **Units**: None                              | counts.  Options: [ counts  ;  |                |
       |                                              | nanotesla ]                    |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

Example Magnetic Channel JSON
-----------------------------

::

   {    "magnetic": {
           "comments": null,
           "component": "Bz",
           "data_logger": {
               "channel_number": 2},
           "data_quality": {
               "warning": "periodic pipeline",
               "rating": {
                   "author": "M. Tee",
                   "method": "Machine Learning",
                   "value": 3}},
		   "fdsn": {
                "channel_code": "LQN", 		   
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

Filter Attributes
------------------ 

:navy:`type`
~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **type**                                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Filter type. Options: [look up | lookup         |
       |                                              |  ;  poles zeros  ;  converter  |                |
       | **Units**: None                              |  ;  FIR  ;  ...]               |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`name`
~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **name**                                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Unique name for the filter     | counts2mv      |
       |                                              | such that it is easy to query. |                |
       | **Units**: None                              | See above for some examples.   |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Alpha Numeric                     |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`units_in`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **units_in**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | The input units for the        | counts         |
       |                                              | filter. Should be SI units or  |                |
       | **Units**: None                              | counts.                        |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`units_out`
~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **units_out**                                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | The output units for the       | millivolts     |
       |                                              | filter. Should be SI units or  |                |
       | **Units**: None                              | counts.                        |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`calibration_date`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **calibration_date**                         | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | If the filter is a calibration | 2010-01-01     |
       |                                              |                                | T00:00:00      |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Date Time                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
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

Auxiliary Channel Attributes
----------------------------- 

:navy:`channel_number`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **channel_number**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Channel Number on the data     | 1              |
       |                                              | logger.                        |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: Integer                            |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`comments`
~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **comments**                                 | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments about the channel | Pc1 at 6pm     |
       |                                              | that would be useful to a      | local time.    |
       | **Units**: None                              | user.                          |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`component`
~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **component**                                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of the component          | temperature    |
       |                                              | measured.  Options: [          |                |
       | **Units**: None                              | temperature  ;  battery  ;     |                |
       |                                              | ... ]                          |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.author`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.author**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Name of person or organization | graduate       |
       |                                              | who rated the data.            | student ace    |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.method`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.method**               | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | The method used to rate the    | standard       |
       |                                              | data.  Should be a descriptive | deviation      |
       | **Units**: None                              | name and not just the name of  |                |
       |                                              | a software package.  If a      |                |
       | **Type**: String                             | rating is provided             |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.rating.value`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.rating.value**                | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Rating from 1-5 where 1 is bad | 4              |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: Integer                            |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`data_quality.warning`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **data_quality.warning**                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any warnings about the data    | periodic       |
       |                                              | that should be noted for       | pipeline noise |
       | **Units**: None                              | users.                         |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	   
:navy:`fdsn.channel_code`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **fdsn.channel_code**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | FDSN channel code, this is a   | LQN            |
       |                                              | 3 character code in the form   |                |
       | **Units**: None                              | [band][type][direction]        |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Alpha-Numeric                     |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+
	   
.. seealso:: https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/ for more information on channel codes.
	   

:navy:`filter.applied`
~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.applied**                           | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Boolean if filter has been     |  [True, False] |
       |                                              | applied or not. If more than   |                |
       | **Units**: None                              | one filter input as a list     |                |
       |                                              | that matches filter.names      |                |
       | **Type**: Boolean                            |                                |                |
       |                                              |                                |                |
       | **Style**: List                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`filter.comments`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.comments**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Any comments on filters that   | low pass is not|
       |                                              | is important for users.        | calibrated     |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`filter.name`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **filter.name**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Name of filter applied or to   | [gain, lp_aux] |
       |                                              | be applied. If more than one   |                |
       | **Units**: None                              | filter input as a list in the  |                |
       |                                              | order in which the should be   |                |
       | **Type**: String                             | applied.                       |                |
       |                                              |                                |                |
       | **Style**: List                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.elevation`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.elevation**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Elevation of channel location  | 123.4          |
       |                                              | in datum specified at survey   |                |
       | **Units**: meters                            | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.latitude`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.latitude**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Latitude of channel location   | 23.134         |
       |                                              | in datum specified at survey   |                |
       | **Units**: decimal degrees                   | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`location.longitude`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **location.longitude**                       | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Longitude of channel location  | 14.23          |
       |                                              | in datum specified at survey   |                |
       | **Units**: decimal degrees                   | level.                         |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`measurement_azimuth`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_azimuth**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Azimuth of channel in the      | 0              |
       |                                              | specified survey.orientation.r |                |
       | **Units**: decimal degrees                   | eference_frame.                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`measurement_tilt`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **measurement_tilt**                         | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Tilt of channel in survey.orie | 0              |
       |                                              | ntation.reference_frame.       |                |
       | **Units**: decimal degrees                   |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`sample_rate`
~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **sample_rate**                              | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Sample rate of the channel.    | 8              |
       |                                              |                                |                |
       | **Units**: samples per second                |                                |                |
       |                                              |                                |                |
       | **Type**: Float                              |                                |                |
       |                                              |                                |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.end`
~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.end**                          | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | End date and time of           | 2020-02-04 T16:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: time                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`time_period.start`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **time_period.start**                        | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Start date and time of         | 2020-02-01 T09:|
       |                                              | collection in UTC.             | 23:45.453670   |
       | **Units**: None                              |                                | +00:00         |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: time                              |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`transformed_azimuth`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_azimuth**                      | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Azimuth angle of channel that  | 0              |
       |                                              | has been transformed into a    |                |
       | **Units**: decimal degrees                   | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | **Type**: Float                              | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`transformed_tilt`
~~~~~~~~~~~~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **transformed_tilt**                         | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :blue:`False`                  | Tilt angle of channel that has | 0              |
       |                                              | been transformed into a        |                |
       | **Units**: decimal degrees                   | specified coordinate system.   |                |
       |                                              | Note this value is only for    |                |
       | **Type**: Float                              | derivative products from the   |                |
       |                                              | archived data.                 |                |
       | **Style**: Number                            |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`type`
~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **type**                                     | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Data type for the channel.     | temperature    |
       |                                              |                                |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Free Form                         |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
       +----------------------------------------------+--------------------------------+----------------+

:navy:`units`
~~~~~~~~~~~~~

.. container::

   .. table::
       :class: tight-table
       :widths: 30 45 15

       +----------------------------------------------+--------------------------------+----------------+
       | **units**                                    | **Description**                | **Example**    |
       +==============================================+================================+================+
       | **Required**: :red:`True`                    | Units of the data.  Options:   | celsius        |
       |                                              | SI units or counts.            |                |
       | **Units**: None                              |                                |                |
       |                                              |                                |                |
       | **Type**: String                             |                                |                |
       |                                              |                                |                |
       | **Style**: Controlled Vocabulary             |                                |                |
       |                                              |                                |                |
       |                                              |                                |                |
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
       <fdsn>
           <channel_code>LQN</channel_code>
       <fdsn>
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

Electromagnetic Frequency Bands
--------------------------------
.. container::
   :name: em

   .. table:: Generalized electromagnetic period bands. Some overlap, use the closest definition.

      +---------------+------------------------------+---------------------------------+
      | **Data Type** | **Definition**               | **Frequency Range**             |
      +===============+==============================+=================================+
      | AMT           | radio magnetotellurics       | :math:`>10^{3}`                 |
      +---------------+------------------------------+---------------------------------+
      | BBMT          | broadband magnetotellurics   | :math:`10^{3}` – :math:`10^{0}` |
      +---------------+------------------------------+---------------------------------+
      | LPMT          | long-period magnetotellurics | :math:`<10^{0}`                 |
      +---------------+------------------------------+---------------------------------+


Channel Components
-------------------
.. container::
   :name: channel_types

   .. table:: These are the common channel components. More can be added.

      ================ ==========================
      **Channel Type** **Definition**
      ================ ==========================
      E                electric field measurement
      B                magnetic field measurement
      T                temperature
      Battery          battery
      SOH              state-of-health
      ================ ==========================

Directions
------------

.. container::
   :name: directions
	
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
