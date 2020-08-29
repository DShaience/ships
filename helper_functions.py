from typing import Tuple, List

import numpy as np
import pandas as pd
import pickle
import os

from matplotlib import pyplot as plt
from pyunpack import Archive
from sklearn import metrics


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


def all_data_files_exists_check(files_list: List[str]):
    """
    :param files_list: list of files to validate
    :return: True if all files exist. False otherwise
    """
    return all([os.path.isfile(file) for file in files_list])


def load_vessels_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :return: extract dataset file (if they don't exist).
    If they exist, pickle them.
    Finally, return the datasets as dataframes
    """
    files_list = [r'data/port_visits_train.csv', r'data/vessels_labels_train.csv', r'data/port_visits_test.csv', r'data/vessels_to_label.csv']
    if not all_data_files_exists_check(files_list):
        print("Extracting RAR archive... ", end="")
        rar_path = r'data/data.rar'
        Archive(rar_path).extractall('data/')
        print("Done. ")

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


def plot_ROC(y_train, y_train_pred_proba, y_holdout, y_holdout_pred_proba, title: str):
    fontsz = 14
    train_fpr, train_tpr, _ = metrics.roc_curve(np.array(y_train), y_train_pred_proba)
    holdout_fpr, holdout_tpr, _ = metrics.roc_curve(np.array(y_holdout), y_holdout_pred_proba)
    plt.figure(figsize=[10, 7])
    train_auc = metrics.auc(train_fpr, train_tpr)
    holdout_auc = metrics.auc(holdout_fpr, holdout_tpr)
    plt.plot(train_fpr, train_tpr, linewidth=4, label='ROC curve Train (area = %0.2f)' % train_auc)
    plt.plot(holdout_fpr, holdout_tpr, linewidth=4, label='ROC curve Holdout (area = %0.2f)' % holdout_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsz + 2)
    plt.ylabel('True Positive Rate', fontsize=fontsz + 2)
    plt.xticks(fontsize=fontsz)
    plt.yticks(fontsize=fontsz)
    plt.title(title, fontsize=fontsz + 4)
    plt.legend(loc="lower right", fontsize=fontsz + 2)
    plt.show()