# -*- coding: utf-8 -*-
"""
Updated on Monday Sep  07
@author: tronan
"""

from mth5 import mth5
import pandas as pd


class MTH5Availability:

    def mth5_availability(mth5_file, run_chan_bool=True):
        """
        Queries for the availability within an input MTH5 file
         :parameter str mth5_file: Full path to MTH5 file
         :parameter bool run_chan_bool: Boolean if output should be
         based on run or channel level
         """
        m = mth5.MTH5()
        m.open_mth5(mth5_file)
        sta_list = m.station_list
        run_df_list = []
        chan_df_list = []
        run_chan_bool = False
        for i in range(len(sta_list)):
            sta_id = sta_list[i]
            run_list = m.get_station(sta_id).groups_list
            for i2 in range(len(run_list)):
                run_id = run_list[i2]
                chan_list = m.stations_group.get_station(sta_id).get_run(run_id).groups_list
                new_run = True
                for i3 in range(len(chan_list)):
                    run_df_templist = []
                    chan_df_templist = []
                    chan_id = chan_list[i3]
                    channel = m.stations_group.get_station(sta_id)\
                               .get_run(run_id).get_channel(chan_id)
                    if run_chan_bool is True:
                        if new_run is True:
                            run_df_templist.append(sta_id)
                            run_df_templist.append(run_id)
                            run_df_templist.append(chan_list)
                            run_df_templist.append(channel.sample_rate)
                            run_df_templist.append(channel.start)
                            run_df_templist.append(channel.end)
                            run_df_list.append(run_df_templist)
                        new_run = False
                    else:
                        chan_df_templist.append(sta_id)
                        chan_df_templist.append(run_id)
                        chan_df_templist.append(chan_id)
                        chan_df_templist.append(channel.sample_rate)
                        chan_df_templist.append(channel.start)
                        chan_df_templist.append(channel.end)
                        chan_df_list.append(chan_df_templist)
        if run_chan_bool is True:
            run_df = pd.DataFrame(run_df_list, columns=['sta', 'run',
                                                        'chan', 'sample_rate',
                                                        'start', 'end'])
           return run_df
        else:
           chan_df  = pd.DataFrame(chan_df_list, columns=['sta', 'run',
                                                          'chan', 'sample_rate',
                                                          'start', 'end'])
           return chan_df
