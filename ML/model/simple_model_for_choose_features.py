#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-26 下午3:25
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

warnings.filterwarnings('ignore')
# my own module
from features import data_utils


def main():
    train, test, macro = data_utils.load_data()

    ylog_train_all = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])

    conbined_data.columns = test.columns.values

    str_columns = conbined_data.select_dtypes(include=['object']).columns.values.tolist()

    # dummy code
    dummies_data = pd.get_dummies(conbined_data[str_columns])
    conbined_data[dummies_data.columns] = dummies_data[dummies_data.columns]
    conbined_data.drop(str_columns, axis=1, inplace=True)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    test_size = (1.0 * test.shape[0]) / (train.shape[0] + test.shape[0])
    print "submit test size:", test_size

    # Convert to numpy values
    X_all = train.values
    print(X_all.shape)

    # Create a validation set, with last 20% of data
    num_train = train.shape[0]
    num_val = int(num_train * 0.2)

    X_train_all = X_all[:num_train]
    X_train = X_all[:num_train - num_val]
    X_val = X_all[num_train - num_val:num_train]
    ylog_train = ylog_train_all[:-num_val]
    ylog_val = ylog_train_all[-num_val:]

    X_test = test

    df_columns = train.columns

    print('X_train_all shape is', X_train_all.shape)
    print('X_train shape is', X_train.shape)
    print('y_train shape is', ylog_train.shape)
    print('X_val shape is', X_val.shape)
    print('y_val shape is', ylog_val.shape)
    print('X_test shape is', X_test.shape)

    dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)
    dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)

    xgb_params = {
        'eta': 0.01,
        'max_depth': 5,
        'subsample': 0.7,
        'booster': 'dart',
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1,
        'seed': 100
    }
    num_round = 500
    xgb_params['nthread'] = 24
    # param['eval_metric'] = "auc"
    plst = xgb_params.items()
    plst += [('eval_metric', 'rmse')]
    evallist = [(dval, 'eval')]

    bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=20, verbose_eval=10)

    num_boost_round = bst.best_iteration + 1
    print 'best_iteration: ', num_boost_round


if __name__ == '__main__':
    main()
