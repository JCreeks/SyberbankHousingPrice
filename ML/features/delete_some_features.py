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
import xgboost as xgb
from sklearn import preprocessing
# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils

def delete_some_features(train, test):

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

    for df in delete_features:
        if df in train.columns.values:
            del train[df]
            del test[df]

    # delete_features = ['school_education_centers_raion_ratio_dis', 'preschool_education_centers_raion_ratio_dis',
    #                    'sport_objects_raion_ratio_dis', 'additional_education_raion_ratio_dis']
    # for df in delete_features:
    #     if df in train.columns.values:
    #         del train[df]
    #         del test[df]

    return train, test


def delete_some_non_important_features(train, test):
    train, test = train.copy(), test.copy()

    train['price_doc'] = np.log1p(train['price_doc'])
    ylog_train_all = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.drop(['timestamp'], axis=1, inplace=True)

    # Deal with categorical values
    for c in conbined_data.columns:
        if conbined_data[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(conbined_data[c].values))
            conbined_data[c] = lbl.transform(list(conbined_data[c].values))

    train = conbined_data.iloc[:train.shape[0], :]

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train, ylog_train_all)
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=400)

    importance = pd.DataFrame.from_dict(model.get_fscore(), orient='index')
    importance.reset_index(inplace=True)
    importance.columns = ['feature', 'fscore']
    importance.sort_values(by='fscore', ascending=False, inplace=True)

    delete_features = importance['feature'][importance['fscore'] < 2]
    return delete_features

def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    # delete_features = delete_some_non_important_features(train, test)
    # print 'delete_features:', len(delete_features)
    # delete_features = delete_features[-20:]
    # print 'delete_features:', len(delete_features)
    #
    # for f in delete_features:
    #     del train[f]
    #     del test[f]

    train, test = delete_some_features(train, test)
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== delete some features =============="
    main()
