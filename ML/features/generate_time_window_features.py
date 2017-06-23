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

    timewindow_features = []
    for timewindow in timewindow_days:
        print 'perform timewindow =', timewindow
        pre_timewindow_salecounts = []
        for i in tqdm(range(conbined_data.shape[0])):
            today_time = conbined_data.loc[i, 'timestamp']
            indexs = (today_time - datetime.timedelta(days=timewindow) < conbined_data['timestamp']) & \
                     (conbined_data['timestamp'] < today_time)
            df = conbined_data[indexs]
            df = df.groupby([target_col]).count()['timestamp'].reset_index()
            df.columns = [target_col, 'sale_count']

            sale_count = df[df[target_col] == conbined_data.loc[i, target_col]]['sale_count'].values
            sale_count = 0 if len(sale_count) == 0 else sale_count[0]
            pre_timewindow_salecounts.append(sale_count)
        feature = 'this_'+target_col+'pre_' + str(timewindow) + '_salecount'
        conbined_data[feature] = pre_timewindow_salecounts
        timewindow_features.append(feature)

    timewindow_salecount_result = conbined_data[timewindow_features]
    return timewindow_salecount_result

def perform_time_window(conbined_data):
    """应用时间窗"""
    # 时间窗大小
    timewindow_days = [30*6, 30*4, 30*2, 30, 20, 10]

    # target_cols = ['sub_area', 'building_uptown']
    target_cols = ['sub_area']
    for target_col in target_cols:
        print '根据 '+ target_col +' 生成时间窗特征......'
        cache_file = Configure.time_window_salecount_features_path.format(target_col)
        if not os.path.exists(cache_file):
            timewindow_salecount_result = generate_timewindow_salecount(conbined_data, timewindow_days, target_col)

            with open(cache_file, "wb") as f:
                cPickle.dump(timewindow_salecount_result, f, -1)
        else:
            with open(cache_file, "rb") as f:
                timewindow_salecount_result = cPickle.load(f)

        conbined_data['index'] = range(conbined_data.shape[0])
        timewindow_salecount_result['index'] = range(timewindow_salecount_result.shape[0])
        conbined_data = pd.merge(conbined_data, timewindow_salecount_result, how='left', on='index')
        del conbined_data['index']

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
    conbined_data.index = range(conbined_data.shape[0])

    conbined_data = perform_time_window(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    #print (test_id)
    test['id'] = test_id.values
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== apply time window generate some statistic features =============="
    main()
