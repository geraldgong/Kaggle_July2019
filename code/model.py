from feature_engineering import PrepareData
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

class RegModel(PrepareData):

    def prepare_train(self):
        train = pd.read_csv(os.path.join(self.dir_out, '_train.csv'))
        test = pd.read_csv(os.path.join(self.dir_out, '_test.csv'))

        drop_list = ['scalar_coupling_constant', 'fc']
        X_train = train.drop(columns=drop_list)
        y = train['scalar_coupling_constant']

        return X_train, test, y

    def model_training(self):
        X_train, X_test, y_train = self.prepare_train()
        # initialize XGBRegressor
        xgb_model = xgb.XGBRegressor(n_estimators=1000, seed=42, n_jobs=4)

        xgb_model.fit(X_train, y_train)
        y_predict_xgb = xgb_model.predict(X_test)
        print('Train score: {}'.format(xgb_model.score(X_train, y_train)))
        # print('Test score: {}'.format(xgb_model.score(X_test, y_test)))

        # # print MAE
        # mae_xgb = mean_absolute_error(np.exp(y_test), np.exp(y_predict_xgb))
        # print('The mean absolute deviation is {}'.format(mae_xgb))
        fig, axes = plt.subplots()
        fig.set_size_inches(16, 12)
        xgb.plot_importance(xgb_model, ax=axes)
        fig.show()

        return y_predict_xgb
if __name__ == "__main__":
    xgb_model = RegModel()
    xgb_model.model_training()