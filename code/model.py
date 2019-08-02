import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from feature_engineering import get_features


def cv_LGBM(X, y):
    # cross-validation
    params = {'num_leaves': 50,
              'min_child_samples': 79,
              'min_data_in_leaf': 100,
              'objective': 'regression',
              'max_depth': 9,
              'learning_rate': 0.2,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.3,
              'colsample_bytree': 1.0,
              }

    reg_model = lgb.LGBMRegressor(**params)
    scores = []
    cv = KFold(n_splits=10, random_state=42)

    for train_index, test_index in cv.split(X):
        print("Train Index: ", train_index)
        print("Test Index: ", test_index, "\n")

        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]
        reg_model.fit(X_train, y_train)
        scores.append(reg_model.score(X_test, y_test))

    print('Avg. score: \n {}'.format(np.mean(scores)))

    ####################################################################################################################
    # Validation

    y_predict = reg_model.predict(X_test)
    print('Train score: {}'.format(reg_model.score(X_train, y_train)))
    print('Test score: {}'.format(reg_model.score(X_test, y_test)))

    fig, axes = plt.subplots()
    fig.set_size_inches(16, 12)
    lgb.plot_importance(reg_model, ax=axes)
    plt.show()

    # print MAE
    error = mean_absolute_error(y_test, y_predict)
    print('The mean absolute deviation is {}'.format(np.log(error)))

    return reg_model

def prediction(reg_model, test):

    y_predict_lgb = reg_model.predict(test)
    submit = pd.read_csv(os.path.join(filepath, 'sample_submission.csv')).drop(columns='scalar_coupling_constant')
    submit['scalar_coupling_constant'] = y_predict_lgb
    submit.to_csv(os.path.join(dir_out, 'submission.csv'), index=False)


if __name__ == "__main__":

    if 'ygong' in os.getcwd():
        filepath = "../data"
        dir_out = "../output"
    else:
        filepath = "/home/gong/Documents/Kaggle_July2019/data"
        dir_out = "/home/gong/Documents/Kaggle_July2019/output"

    train, test = get_features(filepath)

    # drop irrelevant columns
    drop_list = ['id', 'molecule_name', 'atom_0', 'atom_1', 'type']
    X = train.drop(columns=['scalar_coupling_constant'] + drop_list)
    y = train['scalar_coupling_constant']
    test = test.drop(columns=drop_list)

    # cross-validation
    model = cv_LGBM(X, y)
    # prediction on test dataset
    prediction(model, test)

