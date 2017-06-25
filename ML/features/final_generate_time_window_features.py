#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
根据时间窗按照地区统计年、月、季度的销量、平均价格等统计特征
@author: MarkLiu
@time  : 17-6-4 上午11:07
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
# remove warnings
import warnings
import cPickle
from tqdm import tqdm

warnings.filterwarnings('ignore')

import datetime

# my own module
import data_utils
from conf.configure import Configure


def generate_timewindow_salecount(conbined_data_df, timewindow_days, target_col):
    conbined_data = conbined_data_df.copy()
    conbined_data = conbined_data[['id', 'timestamp', target_col]]

    timewindow_salecount_result = pd.DataFrame()
    timewindow_salecount_result['id'] = conbined_data['id']
    for timewindow in timewindow_days:
        print 'perform timewindow =', timewindow
        cache_file = Configure.single_time_window_salecount_features_path.format(target_col, timewindow)

        if not os.path.exists(cache_file):
            pre_timewindow_salecounts = []
            for i in tqdm(range(conbined_data.shape[0])):
                today_time = conbined_data.loc[i, 'timestamp']
                indexs = (today_time - datetime.timedelta(days=timewindow) < conbined_data['timestamp']) & \
                         (conbined_data['timestamp'] < today_time)
                df = conbined_data[indexs]
                if df.shape[0] == 0:
                    pre_timewindow_salecounts.append(0)
                    continue
                df = df.groupby([target_col]).count()['timestamp'].reset_index()
                df.columns = [target_col, 'sale_count']

                sale_count = df[df[target_col] == conbined_data.loc[i, target_col]]['sale_count'].values
                sale_count = 0 if len(sale_count) == 0 else sale_count[0]
                pre_timewindow_salecounts.append(sale_count)

            feature = 'this_' + target_col + 'pre_' + str(timewindow) + '_salecount'
            pre_days_features_df = pd.DataFrame({'id': conbined_data['id'],
                                                 feature: pre_timewindow_salecounts})

            with open(cache_file, "wb") as f:
                cPickle.dump(pre_days_features_df, f, -1)
        else:
            with open(cache_file, "rb") as f:
                pre_days_features_df = cPickle.load(f)

        timewindow_salecount_result = pd.merge(timewindow_salecount_result, pre_days_features_df, how='left', on='id')

    return timewindow_salecount_result


def perform_time_window(conbined_data, timewindow_days):
    """应用时间窗"""
    target_cols = ['full_sq_dis', 'build_year_dis', 'rel_floor']
    for target_col in target_cols:
        if target_col not in conbined_data.columns:
            continue
        print '根据 ' + target_col + ' 生成时间窗特征......'
        timewindow_salecount_result = generate_timewindow_salecount(conbined_data, timewindow_days, target_col)
        conbined_data = pd.merge(conbined_data, timewindow_salecount_result, how='left', on='id')

    return conbined_data


def generate_groupby_timewindow_salecount(conbined_data_df, timewindow_days, target_col):
    conbined_data = conbined_data_df.copy()
    conbined_data = conbined_data[['id', 'timestamp', target_col]]

    timewindow_salecount_result = pd.DataFrame()
    timewindow_salecount_result['id'] = conbined_data['id']
    for timewindow in timewindow_days:
        print 'perform timewindow =', timewindow
        cache_file = Configure.groupby_time_window_salecount_features_path.format(target_col, timewindow)

        if not os.path.exists(cache_file):
            pre_timewindow_salecounts = []
            for i in tqdm(range(conbined_data.shape[0])):
                today_time = conbined_data.loc[i, 'timestamp']
                indexs = (today_time - datetime.timedelta(days=timewindow) < conbined_data['timestamp']) & \
                         (conbined_data['timestamp'] < today_time)
                # 获取时间窗内的数据
                df = conbined_data[indexs]
                df = df.groupby(['sub_area', target_col]).count()['timestamp'].reset_index()
                df.columns = ['sub_area', target_col, 'sale_count']

                sale_count = df[(df['sub_area'] == conbined_data.loc[i, 'sub_area']) and
                                df[target_col] == conbined_data.loc[i, target_col]]['sale_count'].values
                sale_count = 0 if len(sale_count) == 0 else sale_count[0]
                pre_timewindow_salecounts.append(sale_count)
            feature = 'subarea_' + target_col + 'pre_' + str(timewindow) + '_salecount'
            pre_days_features_df = pd.DataFrame({'id': conbined_data['id'],
                                                 feature: pre_timewindow_salecounts})
            with open(cache_file, "wb") as f:
                cPickle.dump(pre_days_features_df, f, -1)
        else:
            with open(cache_file, "rb") as f:
                pre_days_features_df = cPickle.load(f)

        timewindow_salecount_result = pd.merge(timewindow_salecount_result, pre_days_features_df, how='left', on='id')

    return timewindow_salecount_result

def perform_groupby_time_window(conbined_data, timewindow_days):
    """应用时间窗"""
    target_cols = []
    for target_col in target_cols:
        if target_col not in conbined_data.columns:
            continue
        print '根据 ' + target_col + ' groupby 生成时间窗特征......'
        timewindow_salecount_result = generate_groupby_timewindow_salecount(conbined_data, timewindow_days, target_col)
        conbined_data = pd.merge(conbined_data, timewindow_salecount_result, how='left', on='id')

    return conbined_data


def main():
    print 'loading train and test datas...'
    train, test, _ = data_utils.load_data()
    print 'train:', train.shape, ', test:', test.shape

    train_price_doc = train['price_doc']
    train.drop(['price_doc'], axis=1, inplace=True)

    # 合并训练集和测试集
    conbined_data = pd.concat([train[test.columns.values], test])
    conbined_data.columns = test.columns.values
    conbined_data.index = range(conbined_data.shape[0])

    # 时间窗大小
    timewindow_days = [30 * 12, 30 * 11, 30 * 10, 30 * 9, 30 * 8, 30 * 7, 30 * 6, 30 * 5, 30 * 4, 30 * 3, 30 * 2, 30,
                       20, 10]
    conbined_data = perform_time_window(conbined_data, timewindow_days)
    conbined_data = perform_groupby_time_window(conbined_data, timewindow_days)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['price_doc'] = train_price_doc
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== apply time window generate some statistic features =============="
    main()
