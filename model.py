import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from typing import List
import numpy as np
from dataset_preprocessing import feature_generation_main
import seaborn as sns
from report_function import cm_and_classification_report, feature_importance_estimate
from vessels_classes import Quotes

sns.set(color_codes=True)


def get_model_and_tuned_params(model_name: str):
    """
    :param model_name: model name as string
    :return: this function is a short basic wrapper to enable this script to support multiple
    classifiers with only minor changes. The function returns a model object and it's tune_parameters dictionary.
    """
    if model_name.lower() == 'LogisticRegression'.lower():
        tuned_parameters = [{'C': [0.01, 0.1, 1, 10]}]
        model = LogisticRegression(random_state=90210, solver='liblinear', multi_class='auto', penalty='l1')
    elif model_name.lower() == 'RandomForest'.lower():
        tuned_parameters = [{'n_estimators': [10, 15, 30],
                             'criterion': ['gini'],
                             'max_depth': [3, 4, 5, 6],
                             'min_samples_split': [10, 15],
                             'min_samples_leaf': [10, 15]
                             # 'class_weight': [{0: 1, 1: 2, 2: 3}]
                             }]
        model = RandomForestClassifier(random_state=90210)
    elif model_name.lower() == 'AdaBoost'.lower():
        model = AdaBoostClassifier(DecisionTreeClassifier())
        tuned_parameters = [{"base_estimator__criterion": ["gini"],
                             "base_estimator__splitter": ["best"],
                             'base_estimator__max_depth': [2, 3, 4],
                             'base_estimator__min_samples_leaf': [10],
                             "n_estimators": [10, 15, 20, 30],
                             "random_state": [90210],
                             'learning_rate': [0.001, 0.01, 0.1]
                             }]
    else:
        raise ValueError("Unsupported classifier type. Cowardly aborting")
    return model, tuned_parameters


if __name__ == '__main__':
    fun = Quotes('data/quotes.csv')
    fun.print_quote(add_message="Loading train and test feature objects")

    FORCE_GENERATE_DATASET = True
    path_train_features_obj = r'data/train_features_obj.p'
    path_test_features_obj = r'data/test_features_obj.p'

    if FORCE_GENERATE_DATASET:
        train_dataset_obj, test_dataset_obj = feature_generation_main()
        pickle.dump(train_dataset_obj, open(path_train_features_obj, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_dataset_obj, open(path_test_features_obj, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        train_dataset_obj = pickle.load(open(path_train_features_obj, "rb"))
        path_test_features_obj = pickle.load(open(path_test_features_obj, "rb"))

