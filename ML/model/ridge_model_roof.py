#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
Ridge best model run out of folds

@author: MarkLiu
@time  : 17-6-25 下午3:07
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import numpy as np
import pandas as pd  # remove warnings
import warnings

warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
# my own module
from features import data_utils
from conf.configure import Configure


def main():
    train, test, macro = data_utils.load_data()
    train.fillna(0, inplace=True)
    test.fillna(0)
    mult = .969

    train['price_doc'] = train["price_doc"] * mult + 10
    # train['price_doc'] = np.log1p(train['price_doc'])
    y_train = train['price_doc']
    id_train = train['id']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    id_test = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.drop(['timestamp'], axis=1, inplace=True)
    print "conbined_data:", conbined_data.shape

    # Deal with categorical values
    for c in conbined_data.columns:
        if conbined_data[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(conbined_data[c].values))
            conbined_data[c] = lbl.transform(list(conbined_data[c].values))

    del conbined_data['school_education_centers_raion_ratio_dis']
    del conbined_data['preschool_education_centers_raion_ratio_dis']
    del conbined_data['sport_objects_raion_ratio_dis']
    del conbined_data['additional_education_raion_ratio_dis']
    del conbined_data['0_6_all_vs_preschool_quota_dis']

    scaler = StandardScaler()
    conbined_data = scaler.fit_transform(conbined_data)

    train = conbined_data[:train.shape[0], :]
    test = conbined_data[train.shape[0]:, :]

    test_size = (1.0 * test.shape[0]) / train.shape[0]
    print "submit test size:", test_size

    ntrain = train.shape[0]
    ntest = test.shape[0]
    n_folds = 5
    random_seed = 0

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((n_folds, ntest))

    for i, (train_index, test_index) in enumerate(kfold.split(train)):
        print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
        x_tr = train[train_index]
        y_tr = y_train[train_index]
        x_te = train[test_index]

        alphas = np.array([1,0.5, 0.1, 0.05,0.01, 0.005])
        solverOptions = (['svd', 'cholesky', 'sparse_cg', 'sag'])
        # create and fit a ridge regression model, testing each alpha
        model = Ridge(normalize=True, fit_intercept=True) #We have chosen to just normalize the data by default, you could GridsearchCV this is you wanted
        grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas, solver=solverOptions))
        grid.fit(x_tr, y_tr)

        # summarize the results of the grid search
        print 'best_score', grid.best_score_
        print 'alphas:', grid.best_estimator_.alpha
        print 'solverOptions:', grid.best_estimator_.solver

        model = Ridge(normalize=True, alpha=grid.best_estimator_.alpha, fit_intercept=True,
                      solver=grid.best_estimator_.solver)  # paramters tuned using GridSearchCV
        model.fit(x_tr, y_tr)

        train_rmse = mean_squared_error(y_tr, model.predict(x_tr))
        print 'train_rmse =', np.sqrt(train_rmse)

        oof_train[test_index] = model.predict(x_te)
        oof_test_skf[i, :] = model.predict(test)

    oof_test[:] = oof_test_skf.mean(axis=0)

    # 保存 oof 结果
    train_predict = pd.DataFrame({'id': id_train,
                                  'ridge_oof_predict': oof_train})
    test_predict = pd.DataFrame({'id': id_test,
                                 'ridge_oof_predict': oof_test})

    train_predict.to_csv(Configure.train_cv_result_for_model_stacking.format('ridge', time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))),
                         index=False)
    test_predict.to_csv(Configure.test_cv_result_for_model_stacking.format('ridge', time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))),
                         index=False)

if __name__ == '__main__':
    main()
