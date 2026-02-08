mth5.io.zen.zen_tools
=====================

.. py:module:: mth5.io.zen.zen_tools

.. autoapi-nested-parse::

   Created on Tue Apr 18 15:40:28 2023

   @author: jpeacock



Classes
-------

.. autoapisummary::

   mth5.io.zen.zen_tools.ZenSchedule


Functions
---------

.. autoapisummary::

   mth5.io.zen.zen_tools.get_drives
   mth5.io.zen.zen_tools.get_drive_names
   mth5.io.zen.zen_tools.split_station
   mth5.io.zen.zen_tools.copy_from_sd
   mth5.io.zen.zen_tools.delete_files_from_sd


Module Contents
---------------

.. py:function:: get_drives() -> list[str]

   Get a list of logical drives detected on the machine.

   Note: This only works for Windows.

   :returns: List of drives as letters.
   :rtype: list of str

   .. rubric:: Examples

   >>> get_drives()
   ['C', 'D', 'E']


.. py:function:: get_drive_names() -> dict[str, str] | None

   Get a list of drive names detected assuming the cards are named by box and channel.

   :returns: Keys are the drive letters and values are the drive names. Returns None if no drives are found.
   :rtype: dict of str to str or None

   .. rubric:: Examples

   >>> get_drive_names()
   {'D': 'CH1', 'E': 'CH2'}


.. py:function:: split_station(station: str) -> tuple[str, str]

   Split station name into name and number.

   :param station: Full station name.
   :type station: str

   :returns: Tuple containing the station name and number.
   :rtype: tuple of str

   .. rubric:: Examples

   >>> split_station('MT01')
   ('MT', '01')


.. py:function:: copy_from_sd(station: str, save_path: str | pathlib.Path = Path('d:\\\\Peacock\\\\MTData'), channel_dict: dict[str, str] = {'1': 'HX', '2': 'HY', '3': 'HZ', '4': 'EX', '5': 'EY', '6': 'HZ'}, copy_date: str | None = None, copy_type: str = 'all') -> tuple[list[pathlib.Path], pathlib.Path]

   Copy files from SD cards into a common folder (save_path).

   :param station: Full name of station from which data is being saved.
   :type station: str
   :param save_path: Full path to save data to, by default 'd:\Peacock\MTData'.
   :type save_path: str or Path, optional
   :param channel_dict: Keys are the channel numbers as strings and the values are the component that corresponds to that channel. Values are placed in upper case in the code.
   :type channel_dict: dict of str to str, optional
   :param copy_date: Date to copy from depending on copy_type, in 'YYYY-MM-DD' format.
   :type copy_date: str, optional
   :param copy_type: Type of copy operation:
                     - 'all': Copy all files on the SD card.
                     - 'before': Copy files before and on this date.
                     - 'after': Copy files on and after this date.
                     - 'on': Copy files on this date only.
   :type copy_type: {'all', 'before', 'after', 'on'}, optional

   :returns: List of filenames copied to save_path and the save_path itself.
   :rtype: tuple of list of Path and Path

   .. rubric:: Examples

   >>> copy_from_sd('MT01', save_path=r"/home/mt/survey_1")
   ([Path('/home/mt/survey_1/MT01_2026-01-27_120000_256_HX.Z3D')], Path('/home/mt/survey_1'))


.. py:function:: delete_files_from_sd(delete_date: str | None = None, delete_type: str | None = None, delete_folder: str | pathlib.Path = Path().cwd(), verbose: bool = True) -> list[pathlib.Path]

   Delete files from SD card. If delete_date is not None, anything on this date and before will be deleted.

   :param delete_date: Date to delete files from, in 'YYYY-MM-DD' format.
   :type delete_date: str, optional
   :param delete_type: Type of delete operation:
                       - 'all': Delete all files on SD card.
                       - 'before': Delete files on and before delete_date.
                       - 'after': Delete files on and after delete_date.
                       - 'on': Delete files on delete_date.
   :type delete_type: {'all', 'before', 'after', 'on'}, optional
   :param delete_folder: Full path to a folder where files will be moved to just in case. If None, files will be deleted permanently.
   :type delete_folder: str or Path, optional
   :param verbose: If True, print detailed logs, by default True.
   :type verbose: bool, optional

   :returns: List of deleted files.
   :rtype: list of Path

   .. rubric:: Examples

   >>> delete_files_from_sd(delete_date='2026-01-27', delete_type='before', delete_folder=None)
   [Path('D:/file1.Z3D'), Path('D:/file2.Z3D')]


.. py:class:: ZenSchedule

   Deals with reading, writing, and copying schedules.

   Creates a repeating schedule based on the master_schedule. It will then change the first scheduling action to coincide with the master schedule, such that all deployed boxes will have the same schedule.

   .. attribute:: verbose

      If True, print detailed logs.

      :type: bool

   .. attribute:: sr_dict

      Dictionary of sampling rate values.

      :type: dict of str to str

   .. attribute:: sa_list

      List of schedule actions including time and df.

      :type: list of dict

   .. attribute:: ch_cmp_dict

      Dictionary for channel components with keys being the channel number and values being the channel label.

      :type: dict of str to str

   .. attribute:: ch_num_dict

      Dictionary for channel components with keys being the channel label and values being the channel number.

      :type: dict of str to str

   .. attribute:: dt_format

      Date and time format, default is 'YYYY-MM-DD,hh:mm:ss'.

      :type: str

   .. attribute:: initial_dt

      Initial date, or dummy zero date for scheduling.

      :type: str

   .. attribute:: dt_offset

      Start date and time of schedule in dt_format.

      :type: str

   .. attribute:: df_list

      Sequential list of sampling rates to repeat in schedule.

      :type: tuple of int

   .. attribute:: df_time_list

      Sequential list of time intervals to measure for each corresponding sampling rate.

      :type: tuple of str

   .. attribute:: master_schedule

      The schedule that all data loggers should schedule at. Will tailor the schedule to match the master schedule according to dt_offset.

      :type: list of dict


   .. py:attribute:: verbose
      :value: True



   .. py:attribute:: sr_dict


   .. py:attribute:: sa_list
      :value: []



   .. py:attribute:: ch_cmp_dict


   .. py:attribute:: ch_num_dict


   .. py:attribute:: dt_format
      :value: '%Y-%m-%d,%H:%M:%S'



   .. py:attribute:: initial_dt
      :value: '2000-01-01,00:00:00'



   .. py:attribute:: dt_offset


   .. py:attribute:: df_list
      :value: (4096, 256)



   .. py:attribute:: df_time_list
      :value: ('00:10:00', '07:50:00')



   .. py:attribute:: master_schedule


   .. py:method:: add_time(date_time, add_minutes=0, add_seconds=0, add_hours=0, add_days=0)

      add time to a time string
      assuming date_time is in the format  YYYY-MM-DD,HH:MM:SS



   .. py:method:: make_schedule(df_list, df_length_list, repeat=5, t1_dict=None)

      make a repeated schedule given list of sampling frequencies and
      duration for each.
      Arguments:
      -----------
          **df_list** : list
                       list of sampling frequencies in Hz, note needs to be
                       powers of 2 starting at 256
          **df_length_list** : list
                              list of durations in hh:mm:ss format
          **repeat** : int
                       number of times to repeat the sequence
          **t1_dict** : dictionary
                        dictionary returned from get_schedule_offset
      Returns:
      --------
          **time_list**: list of dictionaries with keys:
                          * 'dt' --> date and time of schedule event
                          * 'df' --> sampling rate for that event



   .. py:method:: get_schedule_offset(time_offset, schedule_time_list)

      gets the offset in time from master schedule list and time_offset so
      that all schedules will record at the same time according to master
      schedule list schedule_time_list
      Attributes:
      -----------
          **time_offset** : hh:mm:ss
                            the time offset given to the zen reciever
          **schedule_time_list** : list
                                  list of actual schedule times returned
                                  from make_schedule
      Returns:
      --------
          **s1** : dictionary
                   dictionary with keys:
                       * 'dt' --> date and time of offset from next schedule
                                  event from schedule_time_list
                       * 'df' --> sampling rate of that event



   .. py:method:: write_schedule_for_gui(zen_start: str | None = None, df_list: list[int] | None = None, df_time_list: list[str] | None = None, repeat: int = 8, gain: int = 0, save_path: str | pathlib.Path | None = None, schedule_fn: str = 'zen_schedule.MTsch', version: int = 4) -> None

      Write a zen schedule file.

      :param zen_start: Start time you want the zen to start collecting data, in UTC time. If None, current time is used.
      :type zen_start: str, optional
      :param df_list: List of sampling rates in Hz.
      :type df_list: list of int, optional
      :param df_time_list: List of time intervals corresponding to df_list in hh:mm:ss format.
      :type df_time_list: list of str, optional
      :param repeat: Number of times to repeat the cycle of df_list, by default 8.
      :type repeat: int, optional
      :param gain: Gain on instrument, 2 raised to this number, by default 0.
      :type gain: int, optional
      :param save_path: Path to save the schedule file, by default current working directory.
      :type save_path: str or Path, optional
      :param schedule_fn: Name of the schedule file, by default 'zen_schedule.MTsch'.
      :type schedule_fn: str, optional
      :param version: Version of the schedule file format, by default 4.
      :type version: int, optional

      :rtype: None



