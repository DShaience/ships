import pandas as pd
import numpy as np
from helper_functions import load_vessels_dataset
import featuretools as ft
from collections import Counter
from sklearn.metrics.pairwise import haversine_distances
from math import radians


def primary_profile_count(primary_entity: pd.Series, label: pd.Series) -> dict:
    """
    :param primary_entity: pandas series containing categorical items
    :param label: labels series
    :return: a special dictionary mapping primary_entity items to Pos, Neg, and Total count
    """
    assert len(primary_entity) == len(label), "Primary entity and label must of of equal length. Cowardly aborting"
    primary_set = primary_entity.unique()
    neg = Counter(primary_entity[label == 0])
    pos = Counter(primary_entity[label == 1])

    primary_profile_count_dict = {primary_key: {"Pos": pos[primary_key] if pos[primary_key] else 0,
                                                "Neg": neg[primary_key] if neg[primary_key] else 0} for primary_key in primary_set}
    for key in primary_profile_count_dict:
        primary_profile_count_dict[key]["Total"] = primary_profile_count_dict[key]["Pos"] + primary_profile_count_dict[key]["Neg"]

    return primary_profile_count_dict


class ProfilesTrainCounter:
    """
    This class calculates and holds various entity profiles. For example self.primary_prof_port_id is a dictionary
    of all port_ids with the respective count of Pos, Neg, and Total
    """
    def __init__(self, df: pd.DataFrame, label_colname: str = 'label'):
        self.primary_prof_port_id = primary_profile_count(df['port_id'], df[label_colname])
        self.primary_prof_port_name = primary_profile_count(df['port_name'], df[label_colname])
        self.primary_prof_country = primary_profile_count(df['country'], df[label_colname])


def port_visits_features(df: pd.DataFrame):
    """
    :param df: either train of test dataframe. This function is used identically on both.
    This means this function CANNOT use label (!)
    :return: the same df, enriched with new features. Also returns the list of features that were added
    May also add interim columns that help calculations. These are not included in the returned features list
    """

if __name__ == '__main__':
    # df_port_visits_train, df_vessels_label_train, df_port_visits_test, df_vessels_to_label = load_vessels_dataset()
    df_port_visits_train, df_vessels_label_train, _, _ = load_vessels_dataset()  # to avoid using unfortunate typos for now

    # Adding label
    df_port_visits_train = pd.merge(df_port_visits_train, df_vessels_label_train, how='left', left_on='ves_id', right_on='vessel_id')
    # cols = ['ves_id', 'start_time', 'duration_min', 'port_id', 'country', 'Lat', 'Long', 'port_name', 'vessel_id', 'type', 'label']

    profiles_train = ProfilesTrainCounter(df_port_visits_train, label_colname='label')
    #









































    FEATURETOOLS = False
    if FEATURETOOLS:
        # cols = ['ves_id', 'start_time', 'duration_min', 'port_id', 'country', 'Lat', 'Long', 'port_name', 'vessel_id', 'type', 'label']
        es = ft.EntitySet(id='port_visits')

        es.entity_from_dataframe(entity_id='data', dataframe=df_port_visits_train,
                                 variable_types={
                                     'vessel_id': ft.variable_types.Categorical,
                                     'start_time': ft.variable_types.Datetime,
                                     'duration_min': ft.variable_types.Numeric,
                                     'port_id': ft.variable_types.Categorical,
                                     'country': ft.variable_types.Categorical,
                                     'Lat': ft.variable_types.Numeric,
                                     'Long': ft.variable_types.Numeric,
                                     'port_name': ft.variable_types.Categorical,
                                     'type': ft.variable_types.Categorical,
                                     'label': ft.variable_types.Categorical
                                 },
                                 make_index=True, index='index',
                                 time_index='start_time'
                                 )

        es.normalize_entity(new_entity_id="vessels",
                            base_entity_id="data",
                            index="vessel_id")
        print(es.entity_dict)

        feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='data',
                                              trans_primitives=[#'add_numeric', 'multiply_numeric',
                                                                # 'negate', 'absolute', 'subtract_numeric',
                                                                'year', 'month', 'week', 'hour'
                                                                ],
                                              agg_primitives=['num_unique'],
                                              groupby_trans_primitives=['diff',
                                                                        'cum_min', 'cum_max', 'cum_mean',
                                                                        'time_since_previous'
                                                                        ],
                                              ignore_variables={
                                                  "data": ['index', 'label']
                                              },
                                              max_depth=2, verbose=1, n_jobs=-1)

        #






































    # np.random.seed(0)  # ensures the same set of random numbers are generated
    # date = ['2019-01-01'] * 3 + ['2019-01-02'] * 3 + ['2019-01-03'] * 3
    # var1, var2 = np.random.randn(9), np.random.randn(9) * 20
    # group = ["group1", "group2", "group3"] * 3  # to assign the groups for the multiple group case
    #
    # df_manygrp = pd.DataFrame({"date": date, "group": group, "var1": var1})  # one var, many groups
    # df_combo = pd.DataFrame({"date": date, "group": group, "var1": var1, "var2": var2})  # many vars, many groups
    # df_onegrp = df_manygrp[df_manygrp["group"] == "group1"]  # one var, one group
    #
    # for d in [df_onegrp, df_manygrp, df_combo]:  # loop to apply the change to both dfs
    #     d.loc[d.index, "date"] = pd.to_datetime(d['date']).to_list()
    #     print("Column changed to: ", d.date.dtype.name)
    #
    # df_onegrp.set_index(["date"]).shift(1)
    #
    # df = df_manygrp.set_index(["date", "group"])
    # df = df.unstack().shift(1)
    # df = df.stack(dropna=False)
    #
    # df.reset_index().sort_values("group")
