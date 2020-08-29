import pandas as pd
import numpy as np
from helper_functions import load_vessels_dataset
from collections import Counter
from typing import List
from vessels_classes import Quotes


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
        self.agg_features_df = None                             # Aggregated features data-frame (aggregated over vessel-id. See "raw features" below)

        self.raw_features_colnames: List[str] = []              # keeping track of raw features names (in case I'll add more later)
        # raw features are features before final aggregation. For example, |travel distance| is a raw feature.
        # The corresponding aggregated feature will be |average-travel-distance|, which is
        # the average over all distances traveled by the vessel

        raw_features_to_add = self.calc_raw_features()
        self.raw_features_colnames.extend(raw_features_to_add)
        self.raw_features_colnames.append('duration_min')

        # Calc aggregated features
        self.agg_features_df = self.calc_agg_features()

    @staticmethod
    def __add_prefix(df: pd.DataFrame, cols_list: List[str], prefix: str):
        """
        :param df: aggregated features dataframe
        :param cols_list: list of the aggregated features column names (that is, only the feature, not any meta-data)
        :param prefix: prefix to add
        :return: adds prefix to columns cols_list in df
        """
        assert set(list(df)).intersection(set(cols_list)) == set(cols_list), f"Some columns in cols_list don't exist in df. Cowardly aborting. df: {set(list(df))} vs cols: {set(cols_list)}"
        new_names = [(i, f'{prefix}_' + i) for i in cols_list]
        df.rename(columns=dict(new_names), inplace=True)

    def calc_agg_features(self) -> pd.DataFrame:
        """
        Calculated aggregated features from raw-features data
        :return:
        """
        agg_mean = self.df.groupby(['vessel_id'])[self.raw_features_colnames].mean().reset_index()
        agg_std = self.df.groupby(['vessel_id'])[self.raw_features_colnames].std().reset_index()
        # Some STDs are NaN. Replacing them with 0
        [agg_std[feature].fillna(0.0, inplace=True) for feature in self.raw_features_colnames]

        self.__add_prefix(agg_mean, self.raw_features_colnames, 'mean')
        self.__add_prefix(agg_std, self.raw_features_colnames, 'std')

        agg_features_df = pd.merge(agg_mean, agg_std, on=['vessel_id'], how='inner')
        return agg_features_df

    def calc_raw_features(self) -> List[str]:
        """
        :return: returns a list of raw features names
        """
        self.__add_end_time()  # adds the end-time of the vessels stay at the port
        self.df.sort_values(by='start_time', ascending=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        raw_features_to_add = self.__calc_vessel_velocity()
        return raw_features_to_add

    def __add_end_time(self):
        """
        :return: adds the end-time of the vessels stay at the port. This is used to calculate vessel velocity
        """
        self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        self.df['end_time'] = self.df['start_time'] + pd.to_timedelta(self.df['duration_min'], unit='min')

    @staticmethod
    def __calc_haversine_distance_vectorized(lon1, lat1, lon2, lat2) -> float:
        """
        :param lon1:
        :param lat1:
        :param lon2:
        :param lat2:
        :return: calculates the distance between two points on the globe using 2 sets of long/lat coordinates
        """
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        newlon = lon2 - lon1
        newlat = lat2 - lat1

        haver_formula = np.sin(newlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon / 2.0) ** 2

        dist = 2 * np.arcsin(np.sqrt(haver_formula))
        km = 6367 * dist  # 6367 for distance in KM for miles use 3958
        return km

    def __calc_vessel_velocity(self) -> List[str]:
        """
        :return: Warp 9. Engage.
        Creates three raw features: distance, travel-time, and travel velocity
        returns a list of raw features names
        """
        default_val = 0
        # self.df['end_time_prev'] = self.df.groupby('vessel_id')['end_time'].shift()
        self.df['end_time_prev'] = self.df.groupby('vessel_id')['end_time'].shift()
        self.df['Long_prev'] = self.df.groupby('vessel_id')['Long'].shift()
        self.df['Lat_prev'] = self.df.groupby('vessel_id')['Lat'].shift()

        # Shifting creates NA values in the first sample, where there's no value to shift into.
        # Post-shift condition excludes these samples from feature calculation
        post_shift_cond = ~self.df['end_time_prev'].isna()
        # Create and initialize feature columns to 0
        self.df['distance_km'] = 0
        self.df['travel_time_hours'] = 0
        self.df['travel_velocity_kph'] = 0
        self.df.loc[post_shift_cond, 'distance_km'] = self.__calc_haversine_distance_vectorized(self.df.loc[post_shift_cond, 'Long'].values, self.df.loc[post_shift_cond, 'Lat'].values,
                                                                                     self.df.loc[post_shift_cond, 'Long_prev'].values, self.df.loc[post_shift_cond, 'Lat_prev'].values)
        self.df.loc[post_shift_cond, 'travel_time_hours'] = (self.df.loc[post_shift_cond, 'start_time'] - self.df.loc[post_shift_cond, 'end_time_prev']).astype('timedelta64[s]')/3600
        self.df.loc[post_shift_cond, 'travel_velocity_kph'] = self.df.loc[post_shift_cond, 'distance_km'] / self.df.loc[post_shift_cond, 'travel_time_hours']

        return ['distance_km', 'travel_time_hours', 'travel_velocity_kph']


if __name__ == '__main__':
    fun = Quotes(pd.read_csv('data/quotes.csv'))
    fun.print_quote()
    # df_port_visits_train_merge, df_vessels_label_train, df_port_visits_test, df_vessels_to_label = load_vessels_dataset()
    df_port_visits_train, df_vessels_label_train, _, _ = load_vessels_dataset()  # to avoid using unfortunate typos for now

    # Adding label
    df_port_visits_train_merge = pd.merge(df_port_visits_train, df_vessels_label_train, how='left', left_on='vessel_id', right_on='vessel_id')
    # cols = ['ves_id', 'start_time', 'duration_min', 'port_id', 'country', 'Lat', 'Long', 'port_name', 'vessel_id', 'type', 'label']

    # Calculate profiles
    profiles_train = ProfilesTrainCounter(df_port_visits_train_merge, label_colname='label')

    # data-set, raw features and aggregatesd features
    train_features_dataset = DatasetAndFeatures(df_port_visits_train)
    fun.print_quote()






























# [agg_std[feature].fillna(agg_std[feature].median(), inplace=True) for feature in self.raw_features_colnames]
# print(raw_feature)
# self.df.loc[self.df['vessel_id'] == '577433cae8760d63a5c01356', raw_feature]
