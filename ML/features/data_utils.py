#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-24 下午9:05
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import cPickle
import pandas as pd

# my own module
from conf.configure import Configure


def load_for_impute_data():
    """加载数据"""
    if not os.path.exists(Configure.original_imputed_train_path):
        train = pd.read_csv(Configure.original_train_path)
    else:
        train = pd.read_csv(Configure.original_imputed_train_path)

    if not os.path.exists(Configure.original_imputed_test_path):
        test = pd.read_csv(Configure.original_test_path)
    else:
        test = pd.read_csv(Configure.original_imputed_test_path)

    if not os.path.exists(Configure.original_imputed_macro_path):
        macro = pd.read_csv(Configure.original_macro_path)
    else:
        macro = pd.read_csv(Configure.original_imputed_macro_path)

    return train, test, macro


def load_data():
    """加载数据"""
    if not os.path.exists(Configure.processed_train_path):
        train = pd.read_csv(Configure.original_train_path)
    else:
        with open(Configure.processed_train_path, "rb") as f:
            train = cPickle.load(f)

    if not os.path.exists(Configure.processed_test_path):
        test = pd.read_csv(Configure.original_test_path)
    else:
        with open(Configure.processed_test_path, "rb") as f:
            test = cPickle.load(f)

    if not os.path.exists(Configure.processed_macro_path):
        macro = pd.read_csv(Configure.original_macro_path)
    else:
        with open(Configure.processed_macro_path, "rb") as f:
            macro = cPickle.load(f)

    return train, test, macro


def save_data(train, test, macro):
    """保存数据"""
    if train is not None:
        with open(Configure.processed_train_path, "wb") as f:
            cPickle.dump(train, f, -1)

    if test is not None:
        with open(Configure.processed_test_path, "wb") as f:
            cPickle.dump(test, f, -1)

    if macro is not None:
        with open(Configure.processed_macro_path, "wb") as f:
            cPickle.dump(macro, f, -1)


def load_imputed_data():
    """加载填充缺失的数据"""
    train = pd.read_csv(Configure.original_imputed_train_path, parse_dates=['timestamp'])
    test = pd.read_csv(Configure.original_imputed_test_path, parse_dates=['timestamp'])
    macro = pd.read_csv(Configure.original_imputed_macro_path, parse_dates=['timestamp'])

    return train, test, macro


def save_imputed_data(train, test, macro):
    """保存数据"""
    if train is not None:
        train.to_csv(Configure.original_imputed_train_path, index=False)

    if test is not None:
        test.to_csv(Configure.original_imputed_test_path, index=False)

    if macro is not None:
        macro.to_csv(Configure.original_imputed_macro_path, index=False)


def load_longitude_latitude_data():
    """加载 subarea 经纬度数据"""
    longitude_latitude = pd.read_csv(Configure.original_longitude_latitude_path)
    del longitude_latitude['no']
    return longitude_latitude
