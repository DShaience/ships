import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV, GroupShuffleSplit, GroupKFold
from typing import List
import numpy as np
from dataset_preprocessing import feature_generation_main
import seaborn as sns
from datetime import datetime
from helper_functions import load_vessels_dataset
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

    # todo: in the end remove this and make generating features the default, in all scripts
    FORCE_GENERATE_DATASET = False
    path_train_features_obj = r'data/train_features_obj.p'
    path_test_features_obj = r'data/test_features_obj.p'

    _, df_vessels_label_train, _, _ = load_vessels_dataset()

    if FORCE_GENERATE_DATASET:
        train_dataset_obj, test_dataset_obj = feature_generation_main()
        pickle.dump(train_dataset_obj, open(path_train_features_obj, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_dataset_obj, open(path_test_features_obj, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        train_dataset_obj = pickle.load(open(path_train_features_obj, "rb"))
        test_dataset_obj = pickle.load(open(path_test_features_obj, "rb"))

    train_features_with_label = pd.merge(train_dataset_obj.features_data_set, df_vessels_label_train[['vessel_id', 'label']], how='inner', on='vessel_id')
    col_label = 'label'
    col_id = 'vessel_id'
    features_col_names = [col for col in list(train_features_with_label) if col not in [col_id, col_label]]

    feature_importance_df = feature_importance_estimate(train_features_with_label[features_col_names], train_features_with_label[col_label])
    # top_important_features = feature_importance_df['Feature'].values[0:20]  # Top 20 most important features
    print(feature_importance_df.to_string())

    rs_vessels_sampling = np.random.RandomState(90210)  # random-state for vessels sampling
    # Splitting train and test, by group (using random state for reproducibility)
    inds_train, inds_cv = next(GroupShuffleSplit(test_size=.30, n_splits=1, random_state=90210).split(train_features_with_label, groups=train_features_with_label[col_id]))
    df_train = train_features_with_label.iloc[inds_train].copy(deep=True).reset_index(drop=True)
    df_cv = train_features_with_label.iloc[inds_cv].copy(deep=True).reset_index(drop=True)

    # Preparing data for classifier
    X_train = df_train[features_col_names].copy(deep=True)
    X_train_groups = df_train[col_id].values
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_cv_scaled = scaler.transform(df_cv[features_col_names])
    y_train = df_train[col_label].values
    y_cv = df_cv[col_label].values
    print("")

    # K-Fold, using group-k-fold by patient-id
    group_kfold = GroupKFold(n_splits=6)
    cv = list(group_kfold.split(df_train[features_col_names], df_train[col_label], df_train[col_id]))

    # Model and grid-search
    t0 = datetime.now()
    # model, tuned_parameters = get_model_and_tuned_params(model_name='LogisticRegression')
    # model, tuned_parameters = get_model_and_tuned_params(model_name='RandomForest')
    model, tuned_parameters = get_model_and_tuned_params(model_name='AdaBoost')
    gs = GridSearchCV(model, tuned_parameters, scoring='f1_macro', n_jobs=-1, cv=cv, refit=True, verbose=1)
    gs.fit(X_train_scaled, y_train, groups=X_train_groups)
    best_idx = gs.best_index_
    clf = gs.best_estimator_
    print("Time to complete grid-search: %s seconds" % (datetime.now() - t0).total_seconds())
    print(f"\nBest estimator params:\n\tParams: {gs.best_params_}\n\tBest Score: {gs.best_score_}\n")
    print(f"GridSearch mean-test-score: {gs.cv_results_['mean_test_score'][best_idx]}")
    print(f"GridSearch std-test-score: {gs.cv_results_['std_test_score'][best_idx]}")

    # Prediction and voting
    y_pred_cv = clf.predict(X_cv_scaled)
    # Grouping patients together. The final prediction for each patient is by voting (most frequent predicted class)
    # df_per_patient_grouped_final = per_patient_result(df_cv, col_uid, col_target, y_pred_cv)
    cm_and_classification_report(y_train, y_pred_cv, labels=[0, 1])

