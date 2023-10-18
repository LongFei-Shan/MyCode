from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib import pyplot as plt
# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import xgboost
from sklearn.feature_selection import VarianceThreshold


def WrapperSelectFeature(data, target, n_features_to_select=1):
    '''
    包装法选择特征

    :param data: 数据集
    :param target: 标签
    :param n_features_to_select: 选择特征的个数
    :return: 返回特征选择后的索引
    '''
    # 递归特征消除法，返回特征选择后的数据
    # 参数estimator为基模型
    # 参数n_features_to_select为选择的特征个数
    # 参数step为每轮特征消除的个数
    estimator = RandomForestClassifier()
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit_transform(data, target)
    # 选择的特征的索引
    index = selector.get_support(True)
    # 特征的排名
    rank = selector.ranking_
    # 特征的权重
    weight = selector.estimator_.feature_importances_
    return index, rank, weight


def EmbeddingSelectFeature(data, target):
    """
    嵌入法选择特征
    :param data:  数据集
    :param target:  标签
    :return:
    """
    # 使用xgboost选择特征, 任何基于树的方法都可以进行特征选择
    estimator = xgboost.XGBClassifier()
    # 训练模型
    estimator.fit(data, target)
    # 特征排名
    rank = estimator.feature_importances_

    return rank


def VarSelectFeature(data, threshold=0.1):
    '''
    方差选择法选择特征
    :param data: 数据集
    :param threshold: 方差阈值
    :return: 返回特征选择后的索引
    '''
    # 方差选择法，返回值为特征选择后的数据
    # 参数threshold为方差的阈值
    selector = VarianceThreshold(threshold)
    selector.fit_transform(data)
    # 每个特征的方差
    var = selector.variances_
    # 选择的特征的索引
    index = selector.get_support(True)
    return var, index
