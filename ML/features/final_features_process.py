#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
特征的后期处理，包括：
- Feature Discretization
- Dataset standardization
- Vector normalization
- Feature selection （考虑单独模块）
@author: MarkLiu
@time  : 17-6-15 上午11:30
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils


def feature_discretization(conbined_data):
    """连续特征离散化"""
    num_columns = conbined_data.select_dtypes(exclude=['object']).columns.values
    num_columns = num_columns.tolist()
    num_columns.remove('timestamp')

    for column in num_columns:
        set_len = len(set(conbined_data[column]))
        if set_len <= 10:
            continue

        mingap = 0.
        # if (set_len > 10) and (set_len <= 50):
        #     mingap = (conbined_data[column].max() - conbined_data[column].min()) / 5.0
        #
        # if (set_len > 50) and (set_len <= 100):
        #     mingap = (conbined_data[column].max() - conbined_data[column].min()) / 10.0
        #
        # if (set_len > 100) and (set_len <= 200):
        #     mingap = (conbined_data[column].max() - conbined_data[column].min()) / 20.0
        #
        # if (set_len > 200) and (set_len <= 400):
        #     mingap = (conbined_data[column].max() - conbined_data[column].min()) / 40.0
        #
        # if (set_len > 400) and (set_len <= 1000):
        #     mingap = (conbined_data[column].max() - conbined_data[column].min()) / 100.0
        #
        if (set_len > 3000) and (set_len <= 5000):
            mingap = (conbined_data[column].max() - conbined_data[column].min()) / 400.0

        if (set_len > 5000) and (set_len <= 20000):
            mingap = (conbined_data[column].max() - conbined_data[column].min()) / 1000.0

        conbined_data[column + '_dis'] = np.round(conbined_data[column].values / np.maximum(mingap, 1.0))

    return conbined_data


def feature_distribute_scale(conbined_data):
    """
    特征变量为连续值：如果为长尾分布并且考虑使⽤线性模型，
    可以对变量进⾏幂变换或者对数变换
    """
    num_columns = conbined_data.select_dtypes(exclude=['object']).columns.values
    scater_skew_num_columns = num_columns.tolist()
    scater_skew_num_columns.remove('timestamp')
    for column in scater_skew_num_columns:
        # for boolean features, do not scatter and skewed
        if len(set(conbined_data[column])) < 10:
            scater_skew_num_columns.remove(column)

    # Transform the skewed numeric features by taking log(feature + 1).
    # This will make the features more normal.
    skewed = conbined_data[scater_skew_num_columns].apply(lambda x: skew(x.astype(float)))
    skewed = skewed[skewed > 0.75]
    skewed = skewed.index
    print 'skewed features', skewed.shape[0], ' from total ', conbined_data.shape[1], ' features'
    conbined_data[skewed] = np.log1p(conbined_data[skewed])

    scaler = StandardScaler()
    conbined_data[skewed] = scaler.fit_transform(conbined_data[skewed])

    return conbined_data


def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    train_id = train['id']
    train_price_doc = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values

    # conbined_data = feature_distribute_scale(conbined_data)
    conbined_data = feature_discretization(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== final features process =============="
    main()
