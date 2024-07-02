from loguru import logger
from mt_metadata.timeseries import Survey
from mth5.data.paths import SyntheticTestPaths
from mth5.data.make_mth5_from_asc import _add_survey
from mth5.data.make_mth5_from_asc import create_test3_h5
from mth5.mth5 import MTH5
from mth5.timeseries import ChannelTS
from mth5.timeseries import RunTS
from mth5.utils.helpers import add_filters
from mth5.utils.helpers import get_channel_summary
from mth5.utils.helpers import station_in_mth5
from mth5.utils.helpers import survey_in_mth5

def extract_subset(source_file, target_file, subset_df):
    """
    This function is a proof-of-concept of issue 219: exporting a subset

    TODO: add check that subset_df is a subset of source_file
    TODO: add tests for source/target v0.1.0
    TODO: add tests for source/target v0.2.0
    TODO: Consider add tests for source v0.1.0/target v0.2.0
    TODO: Consider add tests for source v0.2.0/target v0.1.0

    Parameters
    ----------
    source_file
    target_file
    subset_df

    Returns
    -------

    """

    groupby = ["survey", "station", "run"]
    m_source = MTH5(source_file)
    m_source.open_mth5()

    m_target = MTH5(target_file, file_version=m_source.file_version)
    m_target.open_mth5()

    groupby = ["survey", "station", "run"]
    logger.info(f"Testing file_version {m_source.file_version}")
    for (survey_id, station_id, run_id), run_df in subset_df.groupby(groupby):
        survey = m_source.get_survey(survey_id)
        assert survey.metadata.id == survey_id
        # Check if survey already in mth5, don't add again (its cleaner but won't actually matter in results)
        if not survey_in_mth5(m_target, survey.metadata.id):
            logger.info(f"Survey {survey_id} not in mth5 -- Adding")
            _add_survey(m_target, survey.metadata)  # could be done using mth5, but need to handle 0.1.0, 0.2.0
        else:
            print(f"Survey {survey_id} already in target mth5")

        # Add filters
        # TODO: make this only get the filters from the relevant channels
        all_filters = []
        if m_source.file_version == "0.1.0":
            filter_names = m_source.filters_group.filter_dict.keys()
            #            for filter_group in m_source.filters_group.groups_list:
            for filter_name in filter_names:
                filter_instance = m_source.filters_group.to_filter_object(filter_name)
                all_filters.append(filter_instance)
            add_filters(m_target, all_filters)
        elif m_source.file_version == "0.2.0":
            msg = "May Need to access v0.2.0 filters different -- get survey group first --"
            raise NotImplementedError(msg)
        # filters_dict = {x: m.filters_group.to_filter_object(x) for x in channel_metadata.filter.name}

        source_station_obj = m_source.get_station(station_id, survey_id)
        if not station_in_mth5(m_target, station_id, survey_id):
            print(f"Need to make station {station_id}")
            target_station_obj = m_target.add_station(station_id, station_metadata=source_station_obj.metadata, survey=survey_id)
        else:
            print(f"station {station_id} already in target mth5")
            target_station_obj = m_target.get_station(station_id, survey=survey_id)

        source_run_obj = m_source.get_run(station_id, run_id, survey=survey_id)
        logger.info(f"source_run_obj: {source_run_obj}")

        # TODO: Some clever logic that identifies when target channels are the whole source run
        # would be nice, but a bute force iteration should work for POC
        target_channels = run_df.component.to_list()
        source_channels = source_run_obj.channel_summary.component.to_list()
        if set(source_channels) == set(target_channels):
            logger.info("channels in source and target are same -- just map whole RunTS ")
            source_runts = source_run_obj.to_runts()
            target_runts = source_runts
        else:
            msg = "there are a lot of edge cases to worry about here -- Help Wanted"
            logger.info(msg)
            # raise NotImplementedError(msg)
            # Code in this case could be klindo like the following:
            ch_list = []
            for comp in run_df.component.to_list():
                source_ch_obj = source_run_obj.get_channel(comp)
                source_chts = source_ch_obj.to_channel_ts()
                target_chts_metadata = source_chts.channel_metadata.copy()
                target_chts = ChannelTS(
                    channel_type=target_chts_metadata.type,
                    data=source_chts.data_array.data,
                    channel_metadata=target_chts_metadata.to_dict(),
                )
                ch_list.append(target_chts)
            target_runts = RunTS(array_list=ch_list)
            target_runts.run_metadata.id = source_run_obj.metadata.id

        # TODO: decorate this with if run_in_mth5(m_target, run_id, station, survey)
        # try:
        #     target_run_group = target_station_obj.get_run(run_id)
        # except MTH5Error:
        #     target_run_group = target_station_obj.add_run(run_id)
        target_run_group = target_station_obj.add_run(run_id)
        target_run_group.from_runts(target_runts)
        # print(m_target)
        # print(survey)
        # print(survey.metadata)
    print("TODO: ADD FILTERS")
    m_source.close_mth5()
    m_target.close_mth5()
    return
