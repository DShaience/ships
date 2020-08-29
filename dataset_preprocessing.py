import pandas as pd
import numpy as np
from helper_functions import load_vessels_dataset
import featuretools as ft
from collections import Counter
from sklearn.metrics.pairwise import haversine_distances
from math import radians



class ProfilesTrainCounter:
    """
    This class calculates and holds various entity profiles. For example self.primary_prof_port_id is a dictionary
    of all port_ids with the respective count of Pos, Neg, and Total
    """
    def __init__(self, df: pd.DataFrame, label_colname: str = 'label'):
        self.primary_prof_port_id = self.__primary_profile_count(df['port_id'], df[label_colname])
        self.primary_prof_port_name = self.__primary_profile_count(df['port_name'], df[label_colname])
        self.primary_prof_country = self.__primary_profile_count(df['country'], df[label_colname])

    @staticmethod
    def __primary_profile_count(primary_entity: pd.Series, label: pd.Series) -> dict:
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


class DatasetAndFeatures:
    """
    This class takes the original port_visits dataframe (Train OR test) and adds feature columns (also adds some interim calculation columns)
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vessel_ids_set = set(df['vessel_id'].unique())     # used to easily iterate over vessel_ids
        self.calc_features()

    def calc_features(self):
        self.__add_end_time()  # adds the end-time of the vessels stay at the port
        self.df.sort_values(by='start_time', ascending=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.calc_vessel_velocity()

    def __add_end_time(self):
        """
        :return: adds the end-time of the vessels stay at the port. This is used to calculate vessel velocity
        """
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['end_time'] = self.df['start_time'] + pd.to_timedelta(self.df['duration_min'], unit='min')

    @staticmethod
    def __calc_haversine_distance_vectorized(lon1, lat1, lon2, lat2) -> float:
        # lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        newlon = lon2 - lon1
        newlat = lat2 - lat1

        haver_formula = np.sin(newlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon / 2.0) ** 2

        dist = 2 * np.arcsin(np.sqrt(haver_formula))
        km = 6367 * dist  # 6367 for distance in KM for miles use 3958
        return km

    def calc_vessel_velocity(self):
        """
        :return: Warp 9. Engage.
        """
        default_val = 0
        # self.df['end_time_prev'] = self.df.groupby('vessel_id')['end_time'].shift()
        self.df['end_time_prev'] = self.df.groupby('vessel_id')['end_time'].shift()
        self.df['Long_prev'] = self.df.groupby('vessel_id')['Long'].shift()
        self.df['Lat_prev'] = self.df.groupby('vessel_id')['Lat'].shift()

        self.df['distance_km'] = 0
        self.df['travel_time_hours'] = 0
        cond = ~self.df['end_time_prev'].isna()
        self.df.loc[cond, 'distance_km'] = self.__calc_haversine_distance_vectorized(self.df.loc[cond, 'Long'].values, self.df.loc[cond, 'Lat'].values,
                                                                                     self.df.loc[cond, 'Long_prev'].values, self.df.loc[cond, 'Lat_prev'].values)
        self.df.loc[cond, 'travel_time_hours'] = (self.df.loc[cond, 'start_time'] - self.df.loc[cond, 'end_time_prev']).astype('timedelta64[s]')/3600


        # self.df[self.df.loc[:, 'travel_time_hours'] < 0]
        # time diff is: start_time - end_time_prev

        # self.df.loc[self.df['vessel_id'] == '56db7083e4b0a9ba750395d2', :]
        # self.df.loc[self.df['vessel_id'] == '56db88d3e4b006198d26506b', :].to_csv('E:/development/blah.csv', index=False)
        # cols_to_inspect = ['Long', 'Lat', 'Long_prev', 'Lat_prev', 'port_name', 'end_time_prev', 'start_time', 'travel_time_hours', 'distance_km']
        # self.df.loc[self.df['vessel_id'] == '56db88d3e4b006198d26506b', cols_to_inspect]


if __name__ == '__main__':
    # df_port_visits_train, df_vessels_label_train, df_port_visits_test, df_vessels_to_label = load_vessels_dataset()
    df_port_visits_train, df_vessels_label_train, _, _ = load_vessels_dataset()  # to avoid using unfortunate typos for now

    # Adding label
    df_port_visits_train = pd.merge(df_port_visits_train, df_vessels_label_train, how='left', left_on='ves_id', right_on='vessel_id')
    # cols = ['ves_id', 'start_time', 'duration_min', 'port_id', 'country', 'Lat', 'Long', 'port_name', 'vessel_id', 'type', 'label']

    profiles_train = ProfilesTrainCounter(df_port_visits_train, label_colname='label')
    #
    # port_visits_features(df_port_visits_train)

    train_features_dataset = DatasetAndFeatures(df_port_visits_train)






































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
