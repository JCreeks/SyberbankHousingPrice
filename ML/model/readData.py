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

def readDate(isLog1p=True):
    train, test, macro = data_utils.load_data()

    mult = .969

    train['price_doc'] = train["price_doc"] * mult + 10
    if (isLog1p):
        train['price_doc'] = np.log1p(train['price_doc'])
    ylog_train_all = train['price_doc']
    id_train = train['id']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    #submit_ids = test['id']
    submit_ids = pd.read_csv('../../input/test.csv')['id']
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
    
    return train, test, ylog_train_all