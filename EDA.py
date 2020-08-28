import pandas as pd
from helper_functions import load_vessels_dataset
from matplotlib import pyplot as plt
import seaborn as sns
from visualization_functions import series_hist
sns.set(color_codes=True)

if __name__ == '__main__':
    df_port_visits_train, df_vessels_label_train, df_port_visits_test, df_vessels_to_label = load_vessels_dataset()

    # adding label
    df_port_visits_train = pd.merge(df_port_visits_train, df_vessels_label_train, how='left', left_on='ves_id', right_on='vessel_id')

    series_hist(df_port_visits_train.loc[df_port_visits_train['label'] == 0, 'duration_min'],
                df_port_visits_train.loc[df_port_visits_train['label'] == 1, 'duration_min'],
                s1_label='neg', s2_label='pos', xlabel='duration_min', title="Duration", toShow=False)

    bplot = sns.boxplot(x="type", y="duration_min", data=df_port_visits_train, linewidth=2)
    bplot.set_yscale("log")
    bplot.set_title("Duration of stay (log scale) by vessel type")
    plt.show()
