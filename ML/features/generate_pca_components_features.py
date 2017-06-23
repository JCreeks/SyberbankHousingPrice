#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-15 下午9:33
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import pandas as pd
from sklearn.decomposition import PCA
# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils

def generate_pca_components(conbined_data, keep_component = 0.5):
    n_components = int(conbined_data.shape[1] * keep_component)
    num_columns = conbined_data.select_dtypes(exclude=['object']).columns.values
    num_columns = num_columns.tolist()
    num_columns.remove('timestamp')
    pca_components = PCA(n_components=n_components).fit_transform(conbined_data[num_columns])
    pca_components = pd.DataFrame(pca_components, columns=['pca_component_'+str(i)
                                                           for i in range(pca_components.shape[1])])
    return pca_components

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

    pca_components = generate_pca_components(conbined_data, keep_component=0.01)

    pca_train = pca_components.iloc[:train.shape[0], :]
    pca_train['id'] = train_id
    pca_test = pca_components.iloc[:train.shape[0], :]
    pca_test['id'] = test_id

    train = conbined_data.iloc[:train.shape[0], :]
    train['id'] = train_id
    test = conbined_data.iloc[train.shape[0]:, :]
    test['id'] = test_id

    train = pd.merge(train, pca_train, how='left', on='id')
    test = pd.merge(test, pca_test, how='left', on='id')

    train['price_doc'] = train_price_doc
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== generate pca components features =============="
    main()
