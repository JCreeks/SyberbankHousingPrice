#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
关于 neighbourhood 的 features
@author: MarkLiu
@time  : 17-5-26 下午3:36
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


def gen_area_features(conbined_data):
    """构造周边面积相关特征"""
    # 人均周边所占面积
    conbined_data['per_raion_person_area'] = conbined_data['area_m'] / np.maximum(conbined_data["raion_popul"].astype("float"), 1.0)
    # 人均绿地面积
    conbined_data['per_raion_person_green_zone_part '] = conbined_data['green_zone_part'] / conbined_data['raion_popul'].astype(float)
    # 人均工业区面积
    conbined_data['per_raion_person_indust_part'] = conbined_data['indust_part'] / conbined_data['raion_popul'].astype(float)

    return conbined_data


def gen_school_features(conbined_data):
    """构造教育相关特征"""
    # 学龄前人口数所占比例
    conbined_data['children_preschool_popul_ratio'] = \
        conbined_data['children_preschool'] / conbined_data['raion_popul'].astype(float)
    # 学前教育机构的座位数所缺失数目
    conbined_data['children_preschool_preschool_quota_gap'] = \
        conbined_data['children_preschool'] - conbined_data['preschool_quota']
    # 学前教育机构座位缺失比例
    conbined_data['children_preschool_preschool_quota_gap_ratio'] = \
        (conbined_data['children_preschool'] - conbined_data['preschool_quota']) / conbined_data['children_preschool'].astype(float)

    # 学龄人口数所占比例
    conbined_data['children_school_popul_ratio'] = \
        conbined_data['children_school'] / conbined_data['raion_popul'].astype(float)
    # 地区中学座位数所缺失数目
    conbined_data['children_school_school_quota_gap'] = \
        conbined_data['children_school'] - conbined_data['school_quota']
    # 地区中学座位数缺失比例
    conbined_data['children_school_school_quota_gap_ratio'] = \
        (conbined_data['children_school'] - conbined_data['school_quota']) / conbined_data['children_school'].astype(float)

    # 教育机构总数
    conbined_data['total_education_counts'] = conbined_data['preschool_education_centers_raion'] + \
        conbined_data['school_education_centers_raion'] + conbined_data['sport_objects_raion'] + conbined_data['additional_education_raion']
    # 高中院校数所占比例
    conbined_data['school_education_centers_raion_ratio'] = \
        conbined_data['school_education_centers_raion'] / conbined_data['total_education_counts'].astype(float)

    # 学前教育机构所占比例
    conbined_data['preschool_education_centers_raion_ratio'] = \
        conbined_data['preschool_education_centers_raion']  / conbined_data['total_education_counts'].astype(float)
    # 高等教育机构所占比例
    conbined_data['sport_objects_raion_ratio'] = \
        conbined_data['sport_objects_raion']  / conbined_data['total_education_counts'].astype(float)
    # 额外教育机构所占比例
    conbined_data['additional_education_raion_ratio'] = \
        conbined_data['additional_education_raion']  / conbined_data['total_education_counts'].astype(float)

    return conbined_data


def generate_hospital_features(conbined_data):
    """构造医院相关特征"""
    # 各区医院病床数目相对人口的比例
    conbined_data['hospital_beds_raion_raion_popul_ratio'] = \
        conbined_data['hospital_beds_raion'] / conbined_data['raion_popul'].astype(float)
    return conbined_data


def generate_population_features(conbined_data):
    """构造全市人口相关特征"""

    # 全市男性人口比例
    conbined_data['male_ratio'] = conbined_data['male_f'] / conbined_data['full_all'].astype(float)
    # 全市女性人口比例
    conbined_data['female_ratio'] = conbined_data['female_f'] / conbined_data['full_all'].astype(float)

    # 小于工作年龄的人口比例
    conbined_data['young_underwork_vs_full_all_ratio'] = conbined_data['young_all'] / conbined_data['full_all'].astype(float)
    # 小于工龄的男性人口比例
    conbined_data['young_male_vs_underwork_ratio'] = conbined_data['young_male'] / np.maximum(conbined_data["young_all"].astype("float"), 1.0)
    # 小于工龄的女性人口比例
    conbined_data['young_female_vs_underwork_ratio'] = conbined_data['young_female'] / conbined_data['young_all'].astype(float)
    # 男性中小于工龄的人口比例

    conbined_data['young_male_vs_malef_ratio'] = conbined_data['young_male'] / conbined_data['male_f'].astype(float)

    # 女性中小于工龄的人口比例
    conbined_data['young_female_vs_femalef_ratio'] = conbined_data['young_female'] / conbined_data['female_f'].astype(float)

    # 劳动年龄人口比例
    conbined_data['work_all_vs_full_all_ratio'] = conbined_data['work_all'] / conbined_data['full_all'].astype(float)
    # 劳动年龄男性人口比例
    conbined_data['work_male_vs_work_all_ratio'] = conbined_data['work_male'] / np.maximum(conbined_data["work_all"].astype("float"), 1.0)
    # 劳动年龄女性人口比例
    conbined_data['work_female_vs_work_all_ratio'] = conbined_data['work_female'] / conbined_data['work_all'].astype(float)
    # 男性中劳动年龄的人口比例

    conbined_data['work_male_vs_malef_ratio'] = conbined_data['work_male'] / conbined_data['male_f'].astype(float)

    # 女性中劳动年龄的人口比例
    conbined_data['work_female_vs_femalef_ratio'] = conbined_data['work_female'] / conbined_data['female_f'].astype(float)

    # 年龄大于工作年龄的人口比例
    conbined_data['ekder_all_vs_full_all_ratio'] = conbined_data['ekder_all'] / conbined_data['full_all'].astype(float)
    # 年龄大于工作年龄的人口男性人口比例
    conbined_data['ekder_male_vs_ekder_all_ratio'] = conbined_data['ekder_male'] / np.maximum(conbined_data["ekder_all"].astype("float"), 1.0)
    # 年龄大于工作年龄的人口女性人口比例
    conbined_data['ekder_female_vs_ekder_all_ratio'] = conbined_data['ekder_female'] / conbined_data['ekder_all'].astype(float)
    # 男性中年龄大于工作年龄的比例

    conbined_data['ekder_male_vs_malef_ratio'] = conbined_data['ekder_male'] / conbined_data['male_f'].astype(float)

    # 女性中年龄大于工作年龄的比例
    conbined_data['ekder_female_vs_femalef_ratio'] = conbined_data['ekder_female'] / conbined_data['female_f'].astype(float)

    return conbined_data


def generate_population_age_features(conbined_data):
    """生成人口年龄分布的相关特征"""
    # 0-6岁人口比例 0.46068
    conbined_data['0_6_all_age_ratio'] = conbined_data['0_6_all'] / np.maximum(conbined_data["full_all"].astype("float"), 1.0)
    # 7-14岁人口比例 0.460679
    conbined_data['7_14_all_age_ratio'] = conbined_data['7_14_all'] / np.maximum(conbined_data["full_all"].astype("float"), 1.0)
    # 0-17岁人口比例 0.460621
    conbined_data['0_17_all_age_ratio'] = conbined_data['0_17_all'] / np.maximum(conbined_data["full_all"].astype("float"), 1.0)
    # 16-29岁人口比例 0.460626
    conbined_data['16_29_all_age_ratio'] = conbined_data['16_29_all'] / np.maximum(conbined_data["full_all"].astype("float"), 1.0)
    # 0-13岁人口比例 0.460666
    conbined_data['0_13_all_age_ratio'] = conbined_data['0_13_all'] / np.maximum(conbined_data["full_all"].astype("float"), 1.0)

    """可能会反映教育压力"""
    # 0-6岁人口到 children_preschool 学前教育人口的比例 0.46068
    conbined_data['0_6_all_vs_children_preschool'] = conbined_data['0_6_all'] / conbined_data['children_preschool'].astype(float)
    # 0-6岁人口占学前教育机构座位数的比例 0.46065
    conbined_data['0_6_all_vs_preschool_quota'] = conbined_data['0_6_all'] / conbined_data['preschool_quota'].astype(float)
    # 7-14岁人口占children_school学龄人口的比例 0.46068
    conbined_data['7_14_all_vs_children_school'] = conbined_data['7_14_all'] / conbined_data['children_school'].astype(float)
    # 7-14岁人口占教育机构座位数的比例 0.46055
    conbined_data['7_14_all_vs_school_quota'] = conbined_data['7_14_all'] / conbined_data['school_quota'].astype(float)

    return conbined_data


def generate_build_features(conbined_data):
    """构造建筑相关的特征"""
    # 建筑建造材料信息 0.460455
    # block buildings 比例
    conbined_data['build_block_ratio'] = \
        conbined_data['build_count_block'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # wood buildings 比例
    conbined_data['build_wood_ratio'] = \
        conbined_data['build_count_wood'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # frame buildings 比例
    conbined_data['build_frame_ratio'] = \
        conbined_data['build_count_frame'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # brick buildings 比例
    conbined_data['build_brick_ratio'] = \
        conbined_data['build_count_brick'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # monolith buildings 比例
    conbined_data['build_monolith_ratio'] = \
        conbined_data['build_count_monolith'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # panel buildings 比例
    conbined_data['build_panel_ratio'] = \
        conbined_data['build_count_panel'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # foam buildings 比例
    conbined_data['build_foam_ratio'] = \
        conbined_data['build_count_foam'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # slag buildings 比例
    conbined_data['build_slag_ratio'] = \
        conbined_data['build_count_slag'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # mix buildings 比例
    conbined_data['build_mix_ratio'] = \
        conbined_data['build_count_mix'] / conbined_data['raion_build_count_with_material_info'].astype(float)

    # 建筑建造年代信息 0.460343
    # 1920 年之前建造的比例
    conbined_data['build_count_before_1920_ratio'] = \
        conbined_data['build_count_before_1920'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # 1921-1945 建造的比例
    conbined_data['build_count_1921-1945_ratio'] = \
        conbined_data['build_count_1921-1945'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # 1946-1970 建造的比例
    conbined_data['build_count_1946-1970_ratio'] = \
        conbined_data['build_count_1946-1970'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # 1971-1995 建造的比例
    conbined_data['build_count_1971-1995_ratio'] = \
        conbined_data['build_count_1971-1995'] / conbined_data['raion_build_count_with_material_info'].astype(float)
    # 1995 年以后建造的比例
    conbined_data['build_count_after_1995_ratio'] = \
        conbined_data['build_count_after_1995'] / conbined_data['raion_build_count_with_material_info'].astype(float)

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

    conbined_data = gen_area_features(conbined_data)
    conbined_data = gen_school_features(conbined_data)
    conbined_data = generate_hospital_features(conbined_data)
    conbined_data = generate_population_features(conbined_data)
    conbined_data = generate_population_age_features(conbined_data)
    conbined_data = generate_build_features(conbined_data)

    train = conbined_data.iloc[:train.shape[0], :]
    test = conbined_data.iloc[train.shape[0]:, :]

    train['id'] = train_id
    train['price_doc'] = train_price_doc
    test['id'] = test_id
    print 'train:', train.shape, ', test:', test.shape
    print("Save data...")
    data_utils.save_data(train, test, _)


if __name__ == '__main__':
    print "============== generate some neighbourhood features =============="
    main()
