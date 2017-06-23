#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-15 下午5:07
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
import data_utils


def get_xgb_imp(xgb_model, feat_names):
    from numpy import array
    imp_vals = xgb_model.get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    return {k:v/total for k,v in imp_dict.items()}

def feature_select(train, keep_top=0.95):
    """特征选择"""
    train['price_doc'] = np.log1p(train['price_doc'])
    ylog_train_all = train['price_doc']
    train.drop(['price_doc'], axis=1, inplace=True)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(train, ylog_train_all)
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                       verbose_eval=20, show_stdv=False)
    num_boost_rounds = len(cv_output)
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
    importance = pd.DataFrame.from_dict(get_xgb_imp(model, train.columns.values), orient='index')
    importance.reset_index(inplace=True)
    importance.columns = ['feature', 'fscore']
    importance.sort_values(by='fscore', ascending=False, inplace=True)

    print 'importance:', importance.shape
    print importance.shape[0], keep_top
    print importance.shape[0] * keep_top
    keep_top_len = int(importance.shape[0] * keep_top)
    print 'keep_top_len:', keep_top_len
    keep_top_features = importance['feature'].tolist()[:keep_top_len]
    print '原始特征数目：', train.shape[1], ', 特征选择后特征数目：', len(keep_top_features)

    return keep_top_features


def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    train_price_doc = train['price_doc']

    num_columns = train.select_dtypes(exclude=['object']).columns.values
    num_columns = num_columns.tolist()
    num_columns.remove('id')
    num_columns.remove('timestamp')

    print 'perform feature selection in %d numerical features...' % train[num_columns].shape[1]
    keep_features = feature_select(train[num_columns], keep_top=0.98)
    print 'after feature selection numerical features', len(keep_features)
    keep_features.append('id')
    keep_features.append('timestamp')

    train = train[keep_features]
    test = test[keep_features]

    train['price_doc'] = train_price_doc

    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== perform features selection =============="
    main()
