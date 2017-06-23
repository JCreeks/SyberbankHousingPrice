#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-24 下午8:49
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
import pandas as pd
from collections import Counter

# my own module
import data_utils
# remove warnings
import warnings

warnings.filterwarnings('ignore')


def impute_categories_missing_data(dataframe, cate_columns):
    for column in cate_columns:
        most_common = Counter(dataframe[column].tolist()).most_common(1)[0][0]
        dataframe.loc[dataframe[column].isnull(), column] = most_common


def simple_filling_missing_data(dataframe, columns, value):
    """ 填充缺失数据 """
    for column in columns:
        if dataframe[column].isnull().sum() > 0:
            dataframe.loc[dataframe[column].isnull(), column] = value


def simple_impute_data_preprocess(train, test):
    """ clean data """
    print 'clean data...'
    bad_index = train[train.life_sq > train.full_sq].index
    train.ix[bad_index, "life_sq"] = np.NaN
    equal_index = [601, 1896, 2791]
    test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
    bad_index = test[test.life_sq > test.full_sq].index
    test.ix[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.life_sq < 5].index
    train.ix[bad_index, "life_sq"] = np.NaN
    bad_index = test[test.life_sq < 5].index
    test.ix[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.full_sq < 5].index
    train.ix[bad_index, "full_sq"] = np.NaN
    bad_index = test[test.full_sq < 5].index
    test.ix[bad_index, "full_sq"] = np.NaN
    kitch_is_build_year = [13117]
    train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
    bad_index = train[train.kitch_sq >= train.life_sq].index
    train.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[test.kitch_sq >= test.life_sq].index
    test.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
    train.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
    test.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
    train.ix[bad_index, "full_sq"] = np.NaN
    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
    test.ix[bad_index, "full_sq"] = np.NaN
    bad_index = train[train.life_sq > 300].index
    train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
    bad_index = test[test.life_sq > 200].index
    test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
    train.product_type.value_counts(normalize=True)
    test.product_type.value_counts(normalize=True)
    bad_index = train[train.build_year < 1500].index
    train.ix[bad_index, "build_year"] = np.NaN
    bad_index = test[test.build_year < 1500].index
    test.ix[bad_index, "build_year"] = np.NaN
    bad_index = train[train.num_room == 0].index
    train.ix[bad_index, "num_room"] = np.NaN
    bad_index = test[test.num_room == 0].index
    test.ix[bad_index, "num_room"] = np.NaN
    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    train.ix[bad_index, "num_room"] = np.NaN
    bad_index = [3174, 7313]
    test.ix[bad_index, "num_room"] = np.NaN
    bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
    train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
    bad_index = train[train.floor == 0].index
    train.ix[bad_index, "floor"] = np.NaN
    bad_index = train[train.max_floor == 0].index
    train.ix[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.max_floor == 0].index
    test.ix[bad_index, "max_floor"] = np.NaN
    bad_index = train[train.floor > train.max_floor].index
    train.ix[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.floor > test.max_floor].index
    test.ix[bad_index, "max_floor"] = np.NaN
    train.floor.describe(percentiles=[0.9999])
    bad_index = [23584]
    train.ix[bad_index, "floor"] = np.NaN
    train.material.value_counts()
    test.material.value_counts()
    train.state.value_counts()
    bad_index = train[train.state == 33].index
    train.ix[bad_index, "state"] = np.NaN
    test.state.value_counts()

    # brings error down a lot by removing extreme price per sqm
    train.loc[train.full_sq == 0, 'full_sq'] = 50
    train = train[train.price_doc / train.full_sq <= 600000]
    train = train[train.price_doc / train.full_sq >= 10000]

    # 去除train中的价格超过 1e8 area却相对较小的 outlier 数据
    train = train[train['price_doc'] < 1e8]

    simple_filling_missing_data(train, ['state'], -1)
    simple_filling_missing_data(test, ['state'], -1)

    simple_filling_missing_data(train, ['material'], 1)
    simple_filling_missing_data(test, ['material'], 1)

    train['build_year'][train['build_year'] == 20052009] = 2005
    train['build_year'][train['build_year'] == 20] = 2000
    train['build_year'][train['build_year'] == 215] = 2015
    train['build_year'][train['build_year'] == 4965] = 1965
    train['build_year'][train['build_year'] == 71] = 1971

    test['build_year'][test['build_year'] == 20052009] = 2005
    test['build_year'][test['build_year'] == 20] = 2000
    test['build_year'][test['build_year'] == 215] = 2015
    test['build_year'][test['build_year'] == 4965] = 1965
    test['build_year'][test['build_year'] == 71] = 1971

    # train['build_year'][train['build_year'] < 20] = None
    # test['build_year'][test['build_year'] < 20] = None

    return train, test


def impute_categories_data(train, test):
    str_columns = train.select_dtypes(include=['object']).columns.values.tolist()
    impute_categories_missing_data(train, str_columns)
    impute_categories_missing_data(test, str_columns)
    return train, test


def contains_null(dataframe):
    missing_df = dataframe.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df[missing_df.missing_count > 0]
    return missing_df.shape[0] > 0


def impute_macro_data(macro):
    # child_on_acc_pre_school:等待学前教育机构确定的儿童数量
    simple_filling_missing_data(macro, ['child_on_acc_pre_school'], 'None')
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '#!'] = -1
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == 'None'] = -1
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '45,713'] = 45713
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '7,311'] = 7311
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '3,013'] = 3013
    macro['child_on_acc_pre_school'][macro['child_on_acc_pre_school'] == '16,765'] = 16765

    simple_filling_missing_data(macro, ['modern_education_share'], 'None')
    macro['modern_education_share'][macro['modern_education_share'] == 'None'] = -1
    macro['modern_education_share'][macro['modern_education_share'] == '90,92'] = 9092
    macro['modern_education_share'][macro['modern_education_share'] == '93,08'] = 9308
    macro['modern_education_share'][macro['modern_education_share'] == '95,4918'] = 954918

    simple_filling_missing_data(macro, ['old_education_build_share'], 'None')
    macro['old_education_build_share'][macro['old_education_build_share'] == 'None'] = -1
    macro['old_education_build_share'][macro['old_education_build_share'] == '23,14'] = 2314
    macro['old_education_build_share'][macro['old_education_build_share'] == '25,47'] = 2547
    macro['old_education_build_share'][macro['old_education_build_share'] == '8,2517'] = 82517

    # 此处简单处理
    macro.fillna(-1, inplace=True)

    return macro


def impute_missing_data_based_on_distribution(train, test):
    """
    ◇ 特征值为连续值：按不同的分布类型对缺失值进⾏补全：偏正态分布，使⽤均值代替，可以保持数
                   据的均值；偏长尾分布，使⽤中值代替，避免受 outlier 的影响；
    ◇ 特征值为离散值：使⽤众数代替。
    """
    train_id = train['id']
    train_price_doc = train['price_doc']
    train.drop(['id', 'price_doc'], axis=1, inplace=True)
    test_id = test['id']
    test.drop(['id'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values

    # 特征值为连续值：按不同的分布类型对缺失值进⾏补全：偏正态分布，使⽤均值代替，可以保持数
    #              据的均值；偏长尾分布，使⽤中值代替，避免受 outlier 的影响；
    mean_features = ['kitch_sq', 'hospital_beds_raion', 'cafe_avg_price_500', 'cafe_sum_500_max_price_avg',
                     'cafe_sum_500_min_price_avg', 'build_year', 'max_floor', 'life_sq', 'preschool_quota',
                     'school_quota', 'cafe_avg_price_1000', 'cafe_sum_1000_max_price_avg', 'build_count_block',
                     'cafe_sum_1000_min_price_avg', 'build_count_1971-1995', 'build_count_1946-1970',
                     'build_count_after_1995', 'build_count_before_1920', 'build_count_1921-1945',
                     'raion_build_count_with_builddate_info', 'build_count_panel', 'build_count_monolith',
                     'raion_build_count_with_material_info', 'build_count_brick', 'cafe_avg_price_1500',
                     'cafe_sum_1500_max_price_avg', 'cafe_sum_1500_min_price_avg', 'cafe_sum_2000_min_price_avg',
                     'cafe_avg_price_2000', 'cafe_sum_2000_max_price_avg', 'cafe_sum_3000_min_price_avg',
                     'cafe_sum_3000_max_price_avg', 'cafe_avg_price_3000', 'cafe_sum_5000_max_price_avg',
                     'cafe_sum_5000_min_price_avg', 'cafe_avg_price_5000', 'prom_part_5000', 'floor',
                     'ID_railroad_station_walk', 'railroad_station_walk_min', 'railroad_station_walk_km',
                     'metro_km_walk', 'metro_min_walk', 'green_part_2000', 'full_sq']
    for mf in mean_features:
        mean = np.expm1(np.log1p(conbined_data[mf].dropna()).mean())
        simple_filling_missing_data(conbined_data, [mf], mean)

    median_features = ['num_room', 'build_count_foam', 'build_count_slag', 'build_count_mix', 'build_count_frame',
                       'build_count_wood']
    # 特征值为离散值：使⽤众数代替
    for mf in median_features:
        median = np.expm1(np.log1p(conbined_data[mf].dropna()).median())
        simple_filling_missing_data(conbined_data, [mf], median)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id

    return train, test


def main():
    print("Load data...")
    train, test, macro = data_utils.load_for_impute_data()
    print train.shape, test.shape, macro.shape

    if contains_null(train) | contains_null(test) | contains_null(macro):
        print("填充 train, test 缺失数据...")
        train, test = simple_impute_data_preprocess(train, test)
        train, test = impute_categories_data(train, test)
        train, test = impute_missing_data_based_on_distribution(train, test)
        print("填充 macro 缺失数据...")
        macro = impute_macro_data(macro)
        print("缺失数据填充完成")
        print("Save data...")
        print train.shape, test.shape, macro.shape
        data_utils.save_imputed_data(train, test, macro)
    else:
        print "没有缺失数据!"


if __name__ == '__main__':
    print "================== impute missing data =================="
    main()
