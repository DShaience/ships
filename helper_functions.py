from typing import Tuple

import pandas as pd
import pickle
import os


def load_and_pickle_data(file_path_csv: str, verbose: bool = True) -> pd.DataFrame:
    """
    :param file_path_csv: file path for csv input
    :param verbose: prints action taken
    :return: if pickle by the same name exists, load that pickle.
    Otherwise, load csv, pickle, and save it.
    Return dataframe
    """
    file_path_pickle = file_path_csv.replace('.csv', '.p')
    if os.path.isfile(file_path_pickle):
        if verbose:
            print(f"Pickle found. Loading {file_path_pickle} ... ", end="")
        df = pickle.load(open(file_path_pickle, "rb"))
        if verbose:
            print("Done. ", end="")
    else:
        if verbose:
            print(f"Loading raw CSV {file_path_csv} ... ", end="")
        df = pd.read_csv(file_path_csv)
        if verbose:
            print("Done. Dumping to pickle... ", end="")
        pickle.dump(df, open(file_path_pickle, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print("Done. ", end="")

    if verbose:
        print("\tRetuning dataframe")
    return df


def load_vessels_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    return df_port_visits_train, df_vessels_label_train, df_port_visits_test, df_vessels_to_label