import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, mean_squared_error as mse, r2_score

class TwoLevelModelStacking(object):
    """Two layer model stacking"""

    def __init__(self, train, y_train, test,
                 models, stacking_model, metric = mse,
                 stacking_with_pre_features=True, n_folds=5, random_seed=0):
        self.train = train
        self.y_train = y_train
        self.test = test
        self.n_folds = n_folds
        self.models = models
        self.stacking_model = stacking_model 
        self.metric = metric

        # stacking_with_pre_features : whether to use pre_features in 2nd layer
        self.stacking_with_pre_features = stacking_with_pre_features

        self.ntrain = train.shape[0]
        self.ntest = test.shape[0]
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    def run_out_of_folds(self, clf):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.n_folds, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kfold.split(self.train)):
            print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
            x_tr = self.train[train_index]
            y_tr = self.y_train[train_index]
            x_te = self.train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(self.test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def run_stack_predict(self):
        if self.stacking_with_pre_features:
            x_train = self.train
            x_test = self.test
        #else:
        #    x_train = np.empty((self.ntrain, self.train.shape[1]))
        #    x_test = np.empty((self.ntest, self.test.shape[1]))

        # run level-1 out-of-folds
        for model in self.models:
            oof_train, oof_test = self.run_out_of_folds(model)
            print("{}-1stCV: {}".format(model, self.metric(self.y_train, oof_train))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test
        
        # run level-2 stacking
        self.stacking_model.train(x_train, self.y_train)
       

        # stacking predict
        predicts = self.stacking_model.predict(x_test)
        score = self.stacking_model.getScore()
        print("stackingCV: {}".format(score))
        return predicts, score
    
class ThreeLevelModelStacking(object):
    """three layer model stacking"""

    def __init__(self, train, y_train, test, metric = mse,
                 firstLevelModels, secondLevelModels, stacking_model, 
                 stacking_with_pre_features=True, n_folds=5, random_seed=0):
        self.train = train
        self.y_train = y_train
        self.test = test
        self.n_folds = n_folds
        self.metric = metric
        self.models = firstLevelModels
        self.secondLevelModels = secondLevelModels
        self.stacking_model = stacking_model

        # stacking_with_pre_features : whether to use pre_features in 2nd layer
        self.stacking_with_pre_features = stacking_with_pre_features

        self.ntrain = train.shape[0]
        self.ntest = test.shape[0]
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    def run_out_of_folds(self, clf):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.n_folds, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kfold.split(self.train)):
            print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
            x_tr = self.train[train_index]
            y_tr = self.y_train[train_index]
            x_te = self.train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(self.test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def run_stack_predict(self):
        if self.stacking_with_pre_features:
            x_train = self.train
            x_test = self.test

        # run level-1 out-of-folds
        for model in self.models:
            oof_train, oof_test = self.run_out_of_folds(model)
            #print(oof_train)
            print("{}-1stCV: {}".format(model, self.metric(self.y_train, oof_train))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test
        
        #train_to_save = x_train
        #test_to_save = x_test
        
        # run level-2 out-of-folds
        self.train = x_train
        self.test = x_test
            
        for model in self.secondLevelModels:
            oof_train, oof_test = self.run_out_of_folds(model)
            print("{}-2ndCV: {}".format(model, self.metric(self.y_train, oof_train))
            try:
                x_train = np.concatenate((x_train, oof_train), axis=1)
                x_test = np.concatenate((x_test, oof_test), axis=1)
            except:
                x_train = oof_train
                x_test = oof_test
            
        # run level-3 stacking        
        self.stacking_model.train(x_train, self.y_train)
       
        # stacking predict
        predicts = self.stacking_model.predict(x_test)
        score = self.stacking_model.getScore()
        
        #pd.DataFrame(train_to_save).to_csv("1stLayerX_trainIsLog1p_{}.csv".format(score))
        #pd.DataFrame(test_to_save).to_csv("1stLayerX_testIsLog1p_{}.csv".format(score))
        
        print("stackingCV: {}".format(score))
        return predicts, score

