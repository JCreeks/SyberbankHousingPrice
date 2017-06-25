#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-23 上午11:56
"""
import os
import sys

import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np

# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils

def perform_area_features(train, test):
    """ 处理 area 相关字段 """
    # 添加面积比例 (有一定效果，后期离散化ratio应该会有用)
    train['life_sq_ratio'] = train['life_sq'] / np.maximum(train["full_sq"].astype("float"),1.0)
    train['kitch_sq_ratio'] = train['kitch_sq'] / np.maximum(train["full_sq"].astype("float"),1.0)
    train['kitch_sq_vs_life_ratio'] = train['kitch_sq'] / np.maximum(train["life_sq"].astype("float"),1.0)

    test['life_sq_ratio'] = test['life_sq'] / np.maximum(test["full_sq"].astype("float"),1.0)
    test['kitch_sq_ratio'] = test['kitch_sq'] / np.maximum(test["full_sq"].astype("float"),1.0)
    test['kitch_sq_vs_life_ratio'] = test['kitch_sq'] / np.maximum(test["life_sq"].astype("float"),1.0)

    return train, test


def perform_floor_features(train, test):
    """ 处理 floor 相关字段 """
    train['rel_floor'] = train['floor'] / np.maximum(train["max_floor"].astype("float"),1.0)
    test['rel_floor'] = test['floor'] / np.maximum(test["max_floor"].astype("float"),1.0)
    return train, test


def perform_material_features(train, test):
    """ 处理 floor 相关字段 """
    return train, test


def perform_build_year_features(train, test):
    """ 处理 build_year 相关字段 """
    return train, test


def perform_num_room_features(train, test):
    """ 处理 num_room 相关字段 """
    # 对于 num_room 也视为缺失值
    # train.loc[train['num_room'] == 0, 'num_room'] = 2
    # test.loc[test['num_room'] == 0, 'num_room'] = 2

    # 添加每个 living rome 房间的面积
    # train['per_living_room_sq'] = (train['life_sq'] - train['kitch_sq']) / train['num_room']
    # test['per_living_room_sq'] = (test['life_sq'] - train['kitch_sq']) / test['num_room']

    return train, test


def perform_state_features(train, test):
    """ 处理 state 相关字段 """
    return train, test


def perform_product_type_features(train, test):
    """ 处理 product_type 相关字段 """
    return train, test


def perform_timestamp_features(conbined_data):
    """添加时间属性，用于后期应用时间窗"""
    # 注意 year 特征测试集和训练集不一样，会造成模型过拟合！
    # conbined_data['year'] = conbined_data.timestamp.dt.year
    conbined_data['month'] = conbined_data.timestamp.dt.month
    conbined_data['quarter'] = conbined_data.timestamp.dt.quarter
    conbined_data['weekofyear'] = conbined_data.timestamp.dt.weekofyear

    # Remove timestamp column (may overfit the model in train)
    # conbined_data.drop(['timestamp'], axis=1, inplace=True)

    return conbined_data


def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    print 'perform data cleaning and basic feature engineering'

    train, test = perform_area_features(train, test)
    train, test = perform_floor_features(train, test)
    train, test = perform_state_features(train, test)
    train, test = perform_material_features(train, test)
    train, test = perform_build_year_features(train, test)
    train, test = perform_num_room_features(train, test)
    train, test = perform_product_type_features(train, test)

    train_id = train['id']
    train_price_doc = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values

    conbined_data = perform_timestamp_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "================== train test preprocess =================="
    main()
