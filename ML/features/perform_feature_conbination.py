#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-15 下午10:09
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils

def perform_internal_characteristics(conbined_data):
    """对于高度相关的特征进行组合，同时在 delete_some_features.py 中删除其中的一个"""

    # # Population Internal Characteristics
    # conbined_data['conbine_work_people'] = (0.17*conbined_data['work_all'] + 0.18*conbined_data['work_male'] +
    #                                        0.17 * conbined_data['work_female']) / (0.17 + 0.18 + 0.17)
    # conbined_data['conbine_young_people'] = (0.23*conbined_data['young_all'] + 0.23*conbined_data['young_male'] +
    #                                        0.23 * conbined_data['young_female']) / (0.23 * 3)
    # conbined_data['conbine_male_female'] = (0.16*conbined_data['male_f'] + 0.16*conbined_data['female_f']) / (0.16 * 2)


    return conbined_data

def perform_feature_conbination(conbined_data):
    """ 高 feature importance 的特征进行组合"""
    # importance_features = ['full_sq', 'floor', 'life_sq', 'max_floor', 'life_sq_ratio',
    #                        'kitch_sq_ratio', 'full_sq_dis', 'railroad_km', 'mosque_km',
    #                        'church_synagogue_km', 'water_treatment_km', 'theater_km',
    #                        'metro_min_walk', 'rel_floor', 'kindergarten_km', 'metro_km_avto', 'floor_density',
    #                        'metro_min_avto', 'park_km', 'cemetery_km', 'power_transmission_line_km']
    #
    # for f in importance_features:
    #     conbined_data[f + '_build_year'] = conbined_data[f] * conbined_data['build_year']

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

    conbined_data = perform_internal_characteristics(conbined_data)
    conbined_data = perform_feature_conbination(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== perform feature conbination =============="
    main()
