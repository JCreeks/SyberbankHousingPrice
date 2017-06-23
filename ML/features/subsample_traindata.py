#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-6 下午5:08
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# remove warnings
import warnings

warnings.filterwarnings('ignore')

# my own module
import data_utils


def subsample_train(train):
    # Subsampling suggested by raddar
    trainsub = train[train.timestamp < '2015-01-01']
    trainsub = trainsub[trainsub.product_type == "Investment"]

    ind_1m = trainsub[trainsub.price_doc <= 1000000].index
    ind_2m = trainsub[trainsub.price_doc == 2000000].index
    ind_3m = trainsub[trainsub.price_doc == 3000000].index

    train_index = set(train.index.copy())

    for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
        ind_set_cut = ind.difference(set(ind[::gap]))
        train_index = train_index.difference(ind_set_cut)
    train = train.loc[train_index]

    return train


def main():
    print 'loading train datas...'
    train, test, _ = data_utils.load_imputed_data()
    print 'train:', train.shape

    train = subsample_train(train)
    train = train.reset_index()
    del train['index']

    print 'train:', train.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== subsample train data =============="
    main()
