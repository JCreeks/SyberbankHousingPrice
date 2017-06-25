#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
xgboost best model run out of folds
@author: MarkLiu
@time  : 17-6-25 上午10:25
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
import xgboost as xgb
# remove warnings
import warnings
import time

warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
# my own module
from features import data_utils
from conf.configure import Configure


def main():
    train, test, macro = data_utils.load_data()

    mult = .969

    train['price_doc'] = train["price_doc"] * mult + 10
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

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    test_size = (1.0 * test.shape[0]) / train.shape[0]
    print "submit test size:", test_size

    # Convert to numpy values
    train = train.values
    test = test.values

    ntrain = train.shape[0]
    ntest = test.shape[0]
    n_folds = 5
    random_seed = 0

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((n_folds, ntest))

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'nthread': 24
    }
    num_round = 1000

    dtest = xgb.DMatrix(test)

    for i, (train_index, test_index) in enumerate(kfold.split(train)):
        print 'fold-{}: train: {}, test: {}'.format(i, train_index, test_index)
        x_tr = train[train_index]
        y_tr = y_train[train_index]
        x_te = train[test_index]

        dtrain = xgb.DMatrix(x_tr, y_tr)
        dx_te = xgb.DMatrix(x_te)
        cv_output = xgb.cv(dict(xgb_params, silent=1), dtrain, num_boost_round=num_round, early_stopping_rounds=40)
        num_boost_round = len(cv_output)
        print 'best_iteration: ', num_boost_round
        model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_round)
        train_rmse = mean_squared_error(y_tr, model.predict(dtrain))
        print 'train_rmse =', np.sqrt(train_rmse)

        oof_train[test_index] = model.predict(dx_te)
        oof_test_skf[i, :] = model.predict(dtest)

    oof_test[:] = oof_test_skf.mean(axis=0)

    # 保存 oof 结果
    train_predict = pd.DataFrame({'id': id_train,
                                  'xgboost_oof_predict': oof_train})
    test_predict = pd.DataFrame({'id': id_test,
                                 'xgboost_oof_predict': oof_test})

    train_predict.to_csv(Configure.train_cv_result_for_model_stacking.format('xgboost', time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))),
                         index=False)
    test_predict.to_csv(Configure.test_cv_result_for_model_stacking.format('xgboost', time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))),
                         index=False)


if __name__ == '__main__':
    main()
