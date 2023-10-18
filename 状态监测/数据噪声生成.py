#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :数据添加噪声.py
# @Time      :2023/9/29 10:50
# @Author    :LongFei Shan
import numpy as np


class NoiseGenerator:
    def __init__(self, noise_type='gaussian', mean=0, std=0.001, uplimit=0.001, downlimit=0, uplimit_dt=30, downlimit_dt=20):
        """
        :param noise_type:  噪声类型，'gaussian'为高斯噪声，'outlier'为离群点噪声
        :param mean:  均值， float
        :param std:  标准差， float
        :param uplimit:  离群点上限， float
        :param downlimit:  离群点下限， float
        :param uplimit_dt:  每隔多少个数据添加一个离群点, 时间上限
        :param downlimit_dt:  每隔多少个数据添加一个离群点，时间下限
        """
        self.noise_type = noise_type
        self.mean = mean
        self.std = std
        self.uplimit = uplimit
        self.downlimit = downlimit
        self.uplimit_dt = uplimit_dt
        self.downlimit_dt = downlimit_dt

    def __gaussian(self, data, mean, std):
        """
        高斯噪声
        :param data:  数据
        :param mean:  均值， float
        :param std:  标准差， float
        :return:
        """
        # 生成噪声数据
        data = np.array(data).astype(np.float64)
        noise_data = np.random.normal(mean, std, data.shape) + data
        return noise_data

    def __outlier(self, data, uplimit, downlimit, uplimit_dt, downlimit_dt):
        """
        离群点噪声
        :param data:  数据
        :param uplimit:  离群点上限， float
        :param downlimit:  离群点下限， float
        :param uplimit_dt:  每隔多少个数据添加一个离群点, 时间上限
        :param downlimit_dt:  每隔多少个数据添加一个离群点，时间下限
        :return:
        """
        # 生成噪声数据
        data = np.array(data).astype(np.float64)
        # 生成符合均匀分布的随机数
        if data.ndim == 1:
            data = data.ravel()
            for i in range(len(data)):
                # 生成时间
                dt = np.random.randint(downlimit_dt, uplimit_dt)
                if i % dt == 0:
                    data[i] += np.random.uniform(downlimit, uplimit)
        else:
            raise ValueError("数据维度不正确,必须为一维数据")
        return data

    def fit_transform(self, x):
        """
        训练模型并生成噪声数据
        :param x:  数据
        :return:
        """
        if self.noise_type == 'gaussian':
            return self.__gaussian(x, self.mean, self.std)
        elif self.noise_type == 'outlier':
            return self.__outlier(x, self.uplimit, self.downlimit, self.uplimit_dt, self.downlimit_dt)
        else:
            raise ValueError("噪声类型不正确")


import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    normalData = pd.read_csv(r"..\数据\正常数据.txt", sep=';').values[:, 1:].astype(np.float64)
    sfg = NoiseGenerator(noise_type='outlier', mean=0, std=0.001, uplimit=0.001, downlimit=-0.001, uplimit_dt=31, downlimit_dt=30)
    fault = sfg.fit_transform(normalData[:, 0])
    plt.plot(normalData[:, 0])
    plt.plot(fault)
    plt.show()
