from feature_engineering import PrepareData
import pandas as pd
import os
from sklearn.model_selection import KFold

class RegModel(PrepareData):

    def prepare_train(self):
        train = pd.read_csv(os.path.join(self.dir_out, '_train.csv'))
        test = pd.read_csv(os.path.join(self.dir_out, '_test.csv'))

        X_train = self.del_cols(train, ['scalar_coupling_constant','fc'])
        X_test = self.del_cols(test, ['scalar_coupling_constant','fc'])
        y = train['scalar_coupling_constant']

        return X_train, X_test, y


