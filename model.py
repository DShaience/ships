import pickle
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold
from typing import List
import numpy as np
from dataset_preprocessing import feature_generation_main
import seaborn as sns
from datetime import datetime
from helper_functions import load_vessels_dataset
from report_function import cm_and_classification_report, feature_importance_estimate
from vessels_classes import Quotes
from matplotlib import pyplot as plt
sns.set(color_codes=True)


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
                             'base_estimator__max_depth': [3, 4],
                             'base_estimator__min_samples_leaf': [10, 20],
                             "n_estimators": [20, 30],
                             "random_state": [90210],
                             'learning_rate': [0.01, 0.1]
                             }]
    # elif model_name.lower() == 'AdaBoost'.lower():
    #     model = AdaBoostClassifier(DecisionTreeClassifier())
    #     tuned_parameters = [{"base_estimator__criterion": ["gini"],
    #                          "base_estimator__splitter": ["best"],
    #                          'base_estimator__max_depth': [2, 3, 4],
    #                          'base_estimator__min_samples_leaf': [10],
    #                          "n_estimators": [10, 15, 20, 30],
    #                          "random_state": [90210],
    #                          'learning_rate': [0.001, 0.01, 0.1]
    #                          }]
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
    inds_train, inds_holdout = next(GroupShuffleSplit(test_size=.30, n_splits=1, random_state=90210).split(train_features_with_label, groups=train_features_with_label[col_id]))
    df_train = train_features_with_label.iloc[inds_train].copy(deep=True).reset_index(drop=True)
    df_holdout = train_features_with_label.iloc[inds_holdout].copy(deep=True).reset_index(drop=True)

    # Preparing data for classifier
    X_train = df_train[features_col_names].copy(deep=True)
    X_train_groups = df_train[col_id].values
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_holdout_scaled = scaler.transform(df_holdout[features_col_names])
    y_train = df_train[col_label].values
    y_holdout = df_holdout[col_label].values
    print("")

    # K-Fold, using group-k-fold by patient-id
    group_kfold = GroupKFold(n_splits=6)
    cv = list(group_kfold.split(df_train[features_col_names], df_train[col_label], df_train[col_id]))

    # Model and grid-search
    t0 = datetime.now()
    # model, tuned_parameters = get_model_and_tuned_params(model_name='LogisticRegression')
    # model, tuned_parameters = get_model_and_tuned_params(model_name='RandomForest')
    model, tuned_parameters = get_model_and_tuned_params(model_name='AdaBoost')
    gs = GridSearchCV(model, tuned_parameters, scoring='balanced_accuracy', n_jobs=-1, cv=cv, refit=True, verbose=1)
    gs.fit(X_train_scaled, y_train, groups=X_train_groups)
    best_idx = gs.best_index_
    clf = gs.best_estimator_
    print("Time to complete grid-search: %s seconds" % (datetime.now() - t0).total_seconds())
    print(f"\nBest estimator params:\n\tParams: {gs.best_params_}\n\tBest Score: {gs.best_score_}\n")
    print(f"GridSearch mean-test-score: {gs.cv_results_['mean_test_score'][best_idx]}")
    print(f"GridSearch std-test-score: {gs.cv_results_['std_test_score'][best_idx]}")

    # Prediction and voting
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_proba_train = clf.predict_proba(X_train_scaled)[:,1]
    y_pred_holdout = clf.predict(X_holdout_scaled)
    y_pred_proba_holdout = clf.predict_proba(X_holdout_scaled)[:,1]

    cm_and_classification_report(y_train, y_pred_train, labels=[0, 1])
    cm_and_classification_report(y_holdout, y_pred_holdout, labels=[0, 1])

    plot_ROC(y_train, y_pred_proba_train, y_holdout, y_pred_proba_holdout, "ROC Curve train vs holdout")

    """
    Appendices

    ================================================
    Appendix A - Feature Importance
    ================================================
                                 Feature  Importance
    23    Pos_ratio_port_name_unique    0.274931
    15           Pos_ratio_port_name    0.240576
    20    port_name_Pos_count_unique    0.112186
    11             Pos_ratio_country    0.071644
    19      Pos_ratio_country_unique    0.049545
    22  port_name_Total_count_unique    0.033158
    3              mean_duration_min    0.022828
    4                std_distance_km    0.022436
    1         mean_travel_time_hours    0.021336
    16      country_Pos_count_unique    0.020700
    18    country_Total_count_unique    0.018167
    0               mean_distance_km    0.014735
    7               std_duration_min    0.013006
    21    port_name_Neg_count_unique    0.012668
    17      country_Neg_count_unique    0.012492
    12           port_name_Pos_count    0.012203
    5          std_travel_time_hours    0.010434
    13           port_name_Neg_count    0.009147
    10           country_Total_count    0.007420
    9              country_Neg_count    0.006665
    14         port_name_Total_count    0.006250
    8              country_Pos_count    0.004120
    6        std_travel_velocity_kph    0.002427
    2       mean_travel_velocity_kph    0.000928

    
    """