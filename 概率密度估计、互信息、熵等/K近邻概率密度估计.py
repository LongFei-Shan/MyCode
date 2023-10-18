#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :K近邻概率密度估计.py
# @Time      :2023/9/22 13:51
# @Author    :LongFei Shan
import numpy as np
from sklearn.neighbors import NearestNeighbors

class KNNEstimator:
    def __init__(self, k=5, algorithm='auto', metric='minkowski', p=2, n_jobs=-1):
        """
        该方法最终计算的是概率密度形状上的估计，而不是概率密度函数，最终概率密度与坐标轴围成的面积不为1
        :param k: K近邻的K值
        :param algorithm: 近邻算法，可选参数有：auto、ball_tree、kd_tree、brute
        :param metric: 距离度量，可选参数有：euclidean、manhattan、chebyshev、minkowski、wminkowski、seuclidean、mahalanobis
        :param p: 距离度量的参数
        :param n_jobs: 并行数
        """
        self.k = k
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, x):
        """
        :param x: 样本
        :return: None
        """
        self.x = x
        self.knn_model = NearestNeighbors(n_neighbors=self.k, algorithm='auto', metric='minkowski', p=2, n_jobs=-1)
        self.knn_model.fit(self.x.reshape(-1, 1))

    def predict(self, x):
        """
        :param x: 样本
        :return: probability_density, probability_density为概率密度估计值
        """
        # 判断x是否为一维数组，如果是，则报错
        if x.ndim == 1:
            raise ValueError("x的维度需要为>1数组，row为样本数，col为特征数")
        # 计算每个数据点的K近邻距离
        distances, indices = self.knn_model.kneighbors(x)
        # 计算概率密度
        probability_density = self.k/(self.x.shape[0] * distances[:, -1])
        # 计算概率密度与x
        return probability_density
