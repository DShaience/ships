import pandas as pd
import numpy as np
from helper_functions import load_vessels_dataset
import featuretools as ft



if __name__ == '__main__':
    df_port_visits_train, df_vessels_label_train, df_port_visits_test, df_vessels_to_label = load_vessels_dataset()

    # Adding label
    df_port_visits_train = pd.merge(df_port_visits_train, df_vessels_label_train, how='left', left_on='ves_id', right_on='vessel_id')

    # entity_cols = ['index', '_temporary_index_column', 'patient_id', 'timestamp',
    #                'measurement_x', 'measurement_y', 'measurement_z', 'label']

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
