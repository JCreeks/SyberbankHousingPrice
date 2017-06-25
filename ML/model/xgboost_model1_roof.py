#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-18 下午4:38
"""
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
import time
from conf.configure import Configure

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

mult = .969

y_train = train["price_doc"] * mult + 10
id_train = train['id']
id_test = test['id']

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)
# print('best num_boost_rounds = ', len(cv_output))
# num_boost_rounds = len(cv_output) # 382

print 'training...'
num_boost_rounds = 385  # This was the CV output, as earlier version shows
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)

# 保存 oof 结果
print 'predict...'
train_predict = pd.DataFrame({'id': id_train,
                              'xgboost1_oof_predict': model.predict(dtrain)})
test_predict = pd.DataFrame({'id': id_test,
                             'xgboost1_oof_predict': model.predict(dtest)})

train_predict.to_csv(Configure.train_cv_result_for_model_stacking.format('xgboost_model1', time.strftime('%Y-%m-%d_%H:%M:%S',
                                                                                                  time.localtime(
                                                                                                      time.time()))),
                     index=False)
test_predict.to_csv(Configure.test_cv_result_for_model_stacking.format('xgboost_model1', time.strftime('%Y-%m-%d_%H:%M:%S',
                                                                                                time.localtime(
                                                                                                    time.time()))),
                    index=False)
