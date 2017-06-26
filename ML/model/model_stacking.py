#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-24 下午12:01
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

from model_stack.model_wrapper import XgbWrapper, SklearnWrapper
from model_stack.model_stack import TwoLevelModelStacking

# my own module
from features import data_utils
from conf.configure import Configure

train, test, macro = data_utils.load_data()
train.fillna(0, inplace=True)
test.fillna(0)
mult = .969

train['price_doc'] = train["price_doc"] * mult + 10
# train['price_doc'] = np.log1p(train['price_doc'])
y_train = train['price_doc']
id_train = train['id']
train.drop(['id', 'price_doc'], axis=1, inplace=True)
submit_ids = test['id']
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

et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

rd_params = {
    'alpha': 10
}

ls_params = {
    'alpha': 0.005
}

SEED = 0

xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
rd = SklearnWrapper(clf=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(clf=Lasso, seed=SEED, params=ls_params)

level_1_models = [xg, et, rf, rd, ls]
stacking_model = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
#XgbWrapper(seed=SEED, params=xgb_params)

model_stack = TwoLevelModelStacking(train, y_train, test, level_1_models, stacking_model=stacking_model, stacking_with_pre_features=False)
predicts = model_stack.run_stack_predict()

df_sub = pd.DataFrame({'id': submit_ids, 'price_doc': predicts})
#df_sub.to_csv(Configure.submission_path, index=False)
