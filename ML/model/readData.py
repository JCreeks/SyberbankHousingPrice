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
from sklearn.preprocessing import Imputer

# my own module
from features import data_utils

def readData(isLog1p=True):
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

def data_preprocess(X_train,X_test):

    all_data = pd.concat((X_train,X_test))
    
#     lowerClipCol = ['floor_from_top', 'roomsize', 'extra_area', 'age_at_sale']
#     for c in lowerClipCol:
#         all_data[[c]]=all_data[[c]].clip(lower=0)
    
# # #     to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
# # #     all_data = all_data.drop(to_delete,axis=1)

# #     #train["SalePrice"] = np.log1p(train["SalePrice"])
# #     #log transform skewed numeric features
#     numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#     skewed_feats = X_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#     skewed_feats = skewed_feats[skewed_feats > 0.75]
#     skewed_feats = skewed_feats.index
#     all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#     #all_data = pd.get_dummies(all_data)
    #all_data = all_data.fillna(all_data.median())
    
    all_data.replace([np.infty, -np.infty], np.nan)
    imp = Imputer(missing_values=np.nan, strategy='median')
    all_data=pd.DataFrame(imp.fit_transform(all_data), columns=all_data.columns)
    imp = Imputer(missing_values=np.infty, strategy='median')
    all_data=pd.DataFrame(imp.fit_transform(all_data), columns=all_data.columns)

    X_train = all_data[:X_train.shape[0]]
    X_test = all_data[X_train.shape[0]:]

    return X_train,X_test