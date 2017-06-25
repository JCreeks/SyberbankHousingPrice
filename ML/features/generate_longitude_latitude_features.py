#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
根据 subarea 构造经纬度特征
@author: MarkLiu
@time  : 17-6-8 下午3:36
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
from math import radians, sin, cos, asin, sqrt
# my own module
import data_utils


def calc_distance(lat1, lng1, lat2, lng2):
    """根据经纬度计算"""
    radlat1 = radians(lat1)
    radlat2 = radians(lat2)
    a = radlat1 - radlat2
    b = radians(lng1) - radians(lng2)
    s = 2 * asin(sqrt(pow(sin(a / 2), 2) + cos(radlat1) * cos(radlat2) * pow(sin(b / 2), 2)))
    earth_radius = 6378.137
    s = s * earth_radius
    if s < 0:
        return -s
    else:
        return s

def kremlin_distance(data):
    """计算到 kremlin 的距离"""
    kremlin_longitude = 55.752121
    kremlin_latitude = 37.617664
    return calc_distance(data[0], data[1], kremlin_latitude, kremlin_longitude)

def generate_distance_features(conbined_data, longitude_latitude):
    """ 根据经纬度获取距离特征 """
    conbined_data = pd.merge(conbined_data, longitude_latitude, how='left', on='sub_area')
    kremlin_longitude = 55.752121
    kremlin_latitude = 37.617664
    conbined_data['kremlin_distance'] = (conbined_data['latitude'].values - kremlin_latitude) ** 2 + \
                                        (conbined_data['longitude'].values - kremlin_longitude) ** 2
    conbined_data['kremlin_distance'] = np.sqrt(conbined_data['kremlin_distance'].values)
    del conbined_data['kremlin_distance']
    # 保留 latitude 和 longitude 信息
    # del conbined_data['latitude']
    # del conbined_data['longitude']
    return conbined_data


def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    longitude_latitude = data_utils.load_longitude_latitude_data()
    print 'train:', train.shape, ', test:', test.shape

    train_id = train['id']
    train_price_doc = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values

    conbined_data = generate_distance_features(conbined_data, longitude_latitude)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id.values
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== generate longitude latitude features =============="
    main()
