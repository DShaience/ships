import pandas as pd
from helper_functions import load_and_pickle_data
from matplotlib import pyplot as plt
import seaborn as sns
from visualization_functions import measurement_distributions
sns.set(color_codes=True)

if __name__ == '__main__':
    path_port_visits_train = r'data/port_visits_train.csv'
    path_vessels_label_train = r'data/vessels_labels_train.csv'
    path_port_visits_test = r'data/port_visits_test.csv'
    path_vessels_to_label = r'data/vessels_to_label.csv'

    # when the pickle exists, load from pickle. Otherwise, reads the file, pickles, and loads.
    # This saves time on rerunning the script
    df_port_visits_train = load_and_pickle_data(path_port_visits_train)
    df_vessels_label_train = load_and_pickle_data(path_vessels_label_train)
    df_port_visits_test = load_and_pickle_data(path_port_visits_test)
    df_vessels_to_label = load_and_pickle_data(path_vessels_to_label)

    # adding label
    df_port_visits_train = pd.merge(df_port_visits_train, df_vessels_label_train, how='left', left_on='ves_id', right_on='vessel_id')

    measurement_distributions(df_port_visits_train.loc[df_port_visits_train['label'] == 0, 'duration_min'],
                              df_port_visits_train.loc[df_port_visits_train['label'] == 1, 'duration_min'],
                              s1_label='neg', s2_label='pos', xlabel='duration_min', title="Duration", toShow=False)

    bplot = sns.boxplot(x="type", y="duration_min", data=df_port_visits_train, linewidth=2)
    bplot.set_yscale("log")
    bplot.set_title("Duration of stay (log scale) by vessel type")
    plt.show()
