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
            print(f"Pickle found. \n\tLoading {file_path_pickle} ... ", end="")
        df = pickle.load(open(file_path_pickle, "rb"))
        if verbose:
            print("Done")
    else:
        if verbose:
            print(f"Loading raw CSV \n\t{file_path_csv} ... ", end="")
        df = pd.read_csv(file_path_csv)
        if verbose:
            print("Done. Dumping to pickle... ", end="")
        pickle.dump(df, open(file_path_pickle, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print("Done")

    if verbose:
        print("\tRetuning dataframe")
    return df

