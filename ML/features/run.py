#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-5-23 上午11:57
"""
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

cmd = 'rm ../data/processed_*.pkl'
os.system(cmd)

cmd = 'python impute_missing_data.py'
os.system(cmd)

cmd = 'python subsample_traindata.py'
os.system(cmd)

cmd = 'python train_test_preprocess.py'
os.system(cmd)

cmd = 'python generate_neighbourhood_features.py'
os.system(cmd)

cmd = 'python generate_longitude_latitude_features.py'
os.system(cmd)

cmd = 'python generate_time_window_features.py'
os.system(cmd)

cmd = 'python generate_subarea_features.py'
os.system(cmd)

cmd = 'python final_features_process.py'
os.system(cmd)

cmd = 'python perform_feature_conbination.py'
os.system(cmd)

# 添加 pca components 特征，线上爆炸
# cmd = 'python generate_pca_components_features.py'
# os.system(cmd)

# 对于 xgboost 保留所有特征比特征选择后的效果要好
# cmd = 'python features_selection.py'
# os.system(cmd)

cmd = 'python delete_some_features.py'
os.system(cmd)

# cmd = 'python final_generate_time_window_features.py'
# os.system(cmd)

print 'feature engineering done!'
