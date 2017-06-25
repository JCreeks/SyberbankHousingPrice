#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-6-18 下午5:39
"""
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

cmd = 'rm ../result/*.csv'
os.system(cmd)

cmd = 'python xgboost_model4.py'
os.system(cmd)
