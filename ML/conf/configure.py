#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
全局配置文件
@author: MarkLiu
@time  : 17-5-23 上午10:33
"""
import time


class Configure(object):

    original_train_path = '../../input/train.csv'
    original_test_path = '../../input/test.csv'
    original_macro_path = '../../input/macro.csv'
    original_BAD_ADDRESS_FIX_path = '../../input/BAD_ADDRESS_FIX.xlsx'
 
    original_imputed_train_path = '../../input/imputed_train.csv'
    original_imputed_test_path = '../../input/imputed_test.csv'
    original_imputed_macro_path = '../../input/imputed_macro.csv'
    original_longitude_latitude_path = '../../input/Subarea_Longitud_Latitud.csv'
 
    submission_path = '../result/submission_{}.csv'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))
 
    processed_train_path = '../../input/processed_train_data.pkl'
    processed_test_path = '../../input/processed_test_data.pkl'
    processed_macro_path = '../../input/processed_macro_data.pkl'
 
    time_window_salecount_features_path = '../../input/time_window_{}_subarea_salecount_features.pkl'
 
    multicollinearity_features = '../../input/multicollinearity_features.pkl'
 
    conbined_data_price_distance_path = '../../input/conbined_data_price_distance_path.pkl'
