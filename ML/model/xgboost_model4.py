#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
使用基本的特征构建基本的 xgboost 模型
@author: MarkLiu
@time  : 17-5-25 下午9:03
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
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# my own module
from features import data_utils


def main():
    train, test, macro = data_utils.load_data()

    mult = .969

    train['price_doc'] = train["price_doc"] * mult + 10
    # train['price_doc'] = np.log1p(train['price_doc'])
    ylog_train_all = train['price_doc']
    id_train = train['id']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    submit_ids = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    # macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
    #               "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
    #               "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build", "timestamp"]
    # conbined_data = pd.merge_ordered(conbined_data, macro[macro_cols], on='timestamp', how='left')

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
    X_all = train.values

    # Create a validation set, with last 20% of data
    num_train = int(train.shape[0] / (1+test_size))

    X_train_all = X_all
    X_train = X_all[:num_train]
    X_val = X_all[num_train:]
    ylog_train = ylog_train_all[:num_train]
    ylog_val = ylog_train_all[num_train:]
    X_test = test
    print "validate size:", 1.0*X_val.shape[0] / X_train.shape[0]

    df_columns = train.columns

    print('X_train_all shape is', X_train_all.shape)
    print('X_train shape is', X_train.shape)
    print('y_train shape is', ylog_train.shape)
    print('X_val shape is', X_val.shape)
    print('y_val shape is', ylog_val.shape)
    print('X_test shape is', X_test.shape)

    dtrain_all = xgb.DMatrix(X_train_all, ylog_train_all, feature_names=df_columns)
    dtrain = xgb.DMatrix(X_train, ylog_train, feature_names=df_columns)
    dval = xgb.DMatrix(X_val, ylog_val, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    num_round = 1000
    xgb_params['nthread'] = 24
    evallist = [(dval, 'eval')]

    bst = xgb.train(xgb_params, dtrain, num_round, evallist, early_stopping_rounds=40, verbose_eval=10)

    train_rmse = mean_squared_error(ylog_train, bst.predict(dtrain))
    val_rmse = mean_squared_error(ylog_val, bst.predict(dval))
    print 'train_rmse =', np.sqrt(train_rmse), ', val_rmse =', np.sqrt(val_rmse)

    num_boost_round = bst.best_iteration
    print 'best_iteration: ', num_boost_round
    model = xgb.train(dict(xgb_params, silent=1), dtrain_all, num_boost_round=num_boost_round)

    print 'predict submit...'
    y_pred = model.predict(dtest)
    # y_pred = np.exp(ylog_pred) - 1
    df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': y_pred})
    df_sub.to_csv('xgboost_model_4.csv', index=False) # 0.31499

    # save model
    model.save_model('xgboost_model4.model')


if __name__ == '__main__':
    main()
