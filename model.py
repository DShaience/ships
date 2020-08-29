import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold
import numpy as np
from dataset_preprocessing import feature_generation_main
import seaborn as sns
from datetime import datetime
from helper_functions import load_vessels_dataset, plot_ROC
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
                             'base_estimator__max_depth': [3],
                             'base_estimator__min_samples_leaf': [10],
                             'base_estimator__class_weight': [{0: 1, 1: 1}],
                             "n_estimators": [20, 30],
                             "random_state": [90210],
                             'learning_rate': [0.01, 0.05]
                             }]
        # tuned_parameters = [{"base_estimator__criterion": ["gini"],
        #                      "base_estimator__splitter": ["best"],
        #                      'base_estimator__max_depth': [3, 4],
        #                      'base_estimator__min_samples_leaf': [10, 20, 30],
        #                      'base_estimator__class_weight': [{0: 1, 1: 1}, {0: 1, 1: 2}],
        #                      "n_estimators": [20, 30],
        #                      "random_state": [90210],
        #                      'learning_rate': [0.01, 0.05, 0.1]
        #                      }]
    else:
        raise ValueError("Unsupported classifier type. Cowardly aborting")
    return model, tuned_parameters


if __name__ == '__main__':
    print("Loading train and test feature objects")

    path_train_features_obj = r'data/train_features_obj.p'
    path_test_features_obj = r'data/test_features_obj.p'

    _, df_vessels_label_train, _, _ = load_vessels_dataset()

    print("Creating features from datasets")
    train_dataset_obj, test_dataset_obj = feature_generation_main()
    pickle.dump(train_dataset_obj, open(path_train_features_obj, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_dataset_obj, open(path_test_features_obj, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    fun = Quotes('data/quotes.csv')
    fun.print_quote("Finished creating features")

    train_features_with_label = pd.merge(train_dataset_obj.features_data_set, df_vessels_label_train[['vessel_id', 'label']], how='inner', on='vessel_id')
    col_label = 'label'
    col_id = 'vessel_id'
    features_col_names = [col for col in list(train_features_with_label) if col not in [col_id, col_label]]

    feature_importance_df = feature_importance_estimate(train_features_with_label[features_col_names], train_features_with_label[col_label])
    # features_col_names = feature_importance_df['Feature'].values[0:20]  # Top 20 most important features
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
    # Scaling
    X_train_scaled = scaler.transform(X_train)
    X_holdout_scaled = scaler.transform(df_holdout[features_col_names])
    y_train = df_train[col_label].values
    y_holdout = df_holdout[col_label].values

    X_test = test_dataset_obj.features_data_set[features_col_names]
    X_test_scaled = scaler.transform(X_test)
    X_test_groups = test_dataset_obj.features_data_set[col_id].values
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
    y_pred_proba_train = clf.predict_proba(X_train_scaled)[:, 1]
    y_pred_holdout = clf.predict(X_holdout_scaled)
    y_pred_proba_holdout = clf.predict_proba(X_holdout_scaled)[:, 1]

    fun.print_quote(add_message="Classification Report")
    cm_and_classification_report(y_train, y_pred_train, labels=[0, 1])
    cm_and_classification_report(y_holdout, y_pred_holdout, labels=[0, 1])

    print("ROC Curve train vs hold-out")
    plot_ROC(y_train, y_pred_proba_train, y_holdout, y_pred_proba_holdout, "ROC Curve train vs holdout")

    print("Writing final test-set prediction to file")
    y_pred_test = clf.predict(X_test_scaled)
    test_pred_df = pd.DataFrame({"vessel_id": X_test_groups, "y_pred_test": y_pred_test})
    test_pred_df.to_csv('data/vessels_to_label_prediction.csv', index=False)

    fun.print_quote(add_message="Analysis complete. Goodbye")

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


    ================================================
    Appendix B - Classification Report Output
    ================================================
    ==================================================
    Loading train and test feature objects
        "But where is everybody?", -Enrico Fermi
    ==================================================
       
    Fitting 6 folds for each of 72 candidates, totalling 432 fits
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   29.4s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  2.3min
    [Parallel(n_jobs=-1)]: Done 432 out of 432 | elapsed:  5.1min finished
    Time to complete grid-search: 313.176989 seconds
    
    Best estimator params:
        Params: {'base_estimator__class_weight': {0: 1, 1: 2}, 'base_estimator__criterion': 'gini', 'base_estimator__max_depth': 4, 'base_estimator__min_samples_leaf': 30, 
        'base_estimator__splitter': 'best', 'learning_rate': 0.1, 'n_estimators': 30, 'random_state': 90210}
        Best Score: 0.802162148091047
    
    GridSearch mean-test-score: 0.802162148091047
    GridSearch std-test-score: 0.009511447197941073
    
    ================================================================================================================================================
    Classification Report
        "No, I am not dead. Because I refuse to believe the afterlife is run by you. The universe is not so badly designed!", -Jean-Luc Picard
    ================================================================================================================================================
    
         t/p      0     1 
            0 12411   374 
            1   629  1371 
    
                  precision    recall  f1-score   support
    
               0      0.952     0.971     0.961     12785
               1      0.786     0.685     0.732      2000
    
        accuracy                          0.932     14785
       macro avg      0.869     0.828     0.847     14785
    weighted avg      0.929     0.932     0.930     14785
    
    
         t/p      0     1 
            0  5241   198 
            1   302   596 
    
                  precision    recall  f1-score   support
    
               0      0.946     0.964     0.954      5439
               1      0.751     0.664     0.704       898
    
        accuracy                          0.921      6337
       macro avg      0.848     0.814     0.829      6337
    weighted avg      0.918     0.921     0.919      6337
    
    
    
    ==================================================
    ROC Curve train vs hold-out
        "Tea. Earl Grey. Hot.", -Jean-Luc Picard
    ==================================================

    """