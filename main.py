import pandas as pd
import numpy as np
import pickle
import os
from helper_functions import load_and_pickle_data


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

    

