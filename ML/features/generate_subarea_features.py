#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
根据 sub_area 统计各地区的相关特征
@author: MarkLiu
@time  : 17-6-5 下午5:51
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


def generate_subarea_density(conbined_data):
    """统计各地区的相关属性密度"""

    density_features = ['full_sq', 'life_sq', 'floor', 'max_floor',
                        'build_year', 'num_room', 'kitch_sq', 'raion_popul',
                        'green_zone_part', 'children_preschool', 'preschool_quota',
                        'preschool_education_centers_raion', 'children_school',
                        'school_quota', 'school_education_centers_raion',
                        'school_education_centers_top_20_raion', 'hospital_beds_raion',
                        'sport_objects_raion', 'shopping_centers_raion',
                        'office_raion', 'young_all', 'work_all', 'ekder_all',
                        '0_13_female', 'raion_build_count_with_material_info',
                        'build_count_brick', 'raion_build_count_with_builddate_info',
                        'ID_metro', 'green_zone_km', 'industrial_km', 'ID_bus_terminal',
                        'cafe_sum_500_max_price_avg', 'cafe_sum_1000_min_price_avg',
                        'mosque_count_5000', 'sport_count_5000', 'office_count_3000',
                        'trc_count_5000', 'trc_sqm_5000', 'market_count_5000', 'rel_floor',
                        'per_raion_person_area', 'young_male_vs_underwork_ratio']

    conbined_data['area_km'] = conbined_data['area_m'] / 1000000
    for feature in density_features:
        if feature in conbined_data.columns.values:
            conbined_data[feature + '_density'] = conbined_data[feature] / conbined_data['area_km']

    del conbined_data['area_km']
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

    conbined_data = generate_subarea_density(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== generate some sub_area features =============="
    main()
