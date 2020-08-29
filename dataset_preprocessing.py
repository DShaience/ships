import pandas as pd
from helper_functions import load_vessels_dataset
from vessels_classes import Quotes, ProfilesCounter, DatasetAndFeatures


def feature_generation_main():
    fun = Quotes('data/quotes.csv')
    fun.print_quote()
    # df_port_visits_train_merge, df_vessels_label_train, df_port_visits_test, df_vessels_to_label = load_vessels_dataset()
    df_port_visits_train, df_vessels_label_train, _, _ = load_vessels_dataset()  # to avoid using unfortunate typos for now

    # Adding label
    df_port_visits_train_merge = pd.merge(df_port_visits_train, df_vessels_label_train, how='left', left_on='vessel_id', right_on='vessel_id')

    # Calculate profiles
    profiles_train = ProfilesCounter(df_port_visits_train_merge, label_colname='label')

    # data-set, raw features, aggregated features, profile-based features
    train_features_dataset = DatasetAndFeatures(df_port_visits_train, profiles_train)
    fun.print_quote()

    # todo: do feature generation for test also. Write datasets to data/ (as pickles)
    #  then start working on model thank goodness


if __name__ == '__main__':
    feature_generation_main()





























# [agg_std[feature].fillna(agg_std[feature].median(), inplace=True) for feature in self.raw_features_colnames]
# print(raw_feature)
# self.df.loc[self.df['vessel_id'] == '577433cae8760d63a5c01356', raw_feature]
