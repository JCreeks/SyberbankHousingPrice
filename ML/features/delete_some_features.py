#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-16 上午10:55
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import numpy as np
import pandas as pd
# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils

def delete_some_features(conbined_data):

    delete_features = ['per_raion_person_area', 'per_raion_person_green_zone_part', 'per_raion_person_indust_part',
                       'indust_area_m_ratio', 'children_preschool_popul_ratio', 'children_preschool_preschool_quota_gap',
                       'children_preschool_preschool_quota_gap_ratio', 'children_school_popul_ratio', 'children_school_school_quota_gap',
                       'children_school_school_quota_gap_ratio', 'school_education_centers_raion_ratio',
                       'preschool_education_centers_raion_ratio', 'sport_objects_raion_ratio', 'additional_education_raion_ratio',
                       'hospital_beds_raion_raion_popul_ratio', 'male_ratio', 'female_ratio', 'young_underwork_vs_full_all_ratio',
                       'young_male_vs_underwork_ratio', 'young_female_vs_underwork_ratio', 'young_male_vs_malef_ratio',
                       'young_female_vs_femalef_ratio', 'work_all_vs_full_all_ratio', 'work_male_vs_work_all_ratio',
                       'work_female_vs_work_all_ratio', 'work_male_vs_malef_ratio', 'work_female_vs_femalef_ratio',
                       'ekder_all_vs_full_all_ratio', 'ekder_male_vs_ekder_all_ratio', 'ekder_female_vs_ekder_all_ratio',
                       'ekder_male_vs_malef_ratio', 'ekder_female_vs_femalef_ratio', '0_6_all_age_ratio',
                       '7_14_all_age_ratio', '0_17_all_age_ratio', '16_29_all_age_ratio', '0_13_all_age_ratio',
                       '0_6_all_vs_children_preschool', '0_6_all_vs_preschool_quota', '7_14_all_vs_children_school',
                       '7_14_all_vs_school_quota', 'build_block_ratio', 'build_wood_ratio', 'build_frame_ratio',
                       'build_brick_ratio', 'build_monolith_ratio', 'build_panel_ratio', 'build_foam_ratio',
                       'build_slag_ratio', 'build_mix_ratio', 'build_count_before_1920_ratio', 'build_count_1921-1945_ratio',
                       'build_count_1946-1970_ratio', 'build_count_1971-1995_ratio', 'build_count_after_1995_ratio']
    #delete_features = pd.read_table("../../cols_to_drop.txt").iloc[1:62]
    
    #tmp=list(pd.read_table("../../cols_to_drop.txt").iloc[1:62])
    #delete_features = list(set().union(delete_features,tmp))
    del tmp
    
    for df in delete_features:
        if df in conbined_data.columns.values:
            del conbined_data[df]

    # delete_features = ['ID_railroad_station_walk_dis', 'ID_railroad_station_avto_dis', 'ID_big_road1_dis',
    #                    'ID_big_road2_dis', 'ID_railroad_terminal_dis', 'ID_bus_terminal']
    # for df in delete_features:
    #     if df in conbined_data.columns.values:
    #         del conbined_data[df]

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

    conbined_data = delete_some_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== delete some features =============="
    main()
