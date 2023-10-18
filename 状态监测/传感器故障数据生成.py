#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :传感器故障数据生成.py
# @Time      :2023/9/29 10:22
# @Author    :LongFei Shan
import numpy as np


class SensorFaultGenerator:
    def __init__(self, fault_type='bais_fault', fault_value=0.1, dt=1):
        """
        :param fault_type:  故障类型，'bais_fault'为偏置故障，'precision_degradation'为精度衰减故障，'complete_failure'为完全失效故障，'draft'为漂移故障
        :param fault_value:  故障值，float, array-like, shape (n_features,), 对于bais_fault, precision_degradation, complete_failure, draft故障类型，故障值分别为偏置值，相较于正常数据的百分比生成符合正态分布的随机数，恒定输出值，相当于斜率
        :param dt:  每两个数据之间的时间，单位为s，默认为1s，只有draft故障类型需要
        """
        self.fault_type = fault_type
        self.fault_value = fault_value
        self.dt = dt

    def __bais_fault(self, data, value):
        """
        偏置故障
        :param data:  数据
        :param value:  故障偏置值，相较于正常数据的偏置值， float, array-like, shape (n_features,)
        :return:
        """
        # 生成故障数据
        data = np.array(data).astype(np.float64)
        fault_data = data + value
        return fault_data

    def __precision_degradation(self, data, value):
        """
        精度衰减故障
        :param data:  数据
        :param value:  相较于正常数据的百分比生成符合正态分布的随机数， float, array-like, shape (n_features,)
        :return:
        """
        # 生成故障数据
        data = np.array(data).astype(np.float64)
        if data.ndim == 1:
            mean = np.mean(data)
            std = mean * value
            fault_data = np.random.normal(0, std, data.shape) + data
        else:
            mean = np.mean(data, axis=0, keepdims=True)
            std = mean * value
            fault_data = np.random.normal(0, std, data.shape) + data
        return fault_data

    def __complete_failure(self, data, value):
        """
        完全失效故障
        :param data:  数据
        :param value:  恒定输出值， float, array-like, shape (n_features,)
        :return:
        """
        # 生成故障数据
        data = np.array(data)
        fault_data = np.zeros(data.shape)
        fault_data += value
        return fault_data

    def __draft(self, data, value, dt=1):
        """
        漂移故障
        :param data:  数据
        :param value:  漂移值,相当于斜率， float, array-like, shape (n_features,)
        :param dt:  每两个数据之间的时间，单位为s，默认为1s
        :return:
        """
        # 生成故障数据
        data = np.array(data).astype(np.float64)
        if data.ndim == 1:
            time = np.arange(0, data.shape[0] * dt, dt)
        else:
            time = np.arange(0, data.shape[0] * dt, dt).reshape(-1, 1)
        fault_data = data + value * time
        return fault_data

    def fit_transform(self, x):
        """
        生成故障数据
        :param x:  输入数据
        :return: 故障数据
        """
        self.__verify_value(x)
        result = None
        if self.fault_type == 'bais_fault':
            result = self.__bais_fault(x, self.fault_value)
        elif self.fault_type == 'precision_degradation':
            result = self.__precision_degradation(x, self.fault_value)
        elif self.fault_type == 'complete_failure':
            result = self.__complete_failure(x, self.fault_value)
        elif self.fault_type == 'draft':
            result = self.__draft(x, self.fault_value, self.dt)
        else:
            raise ValueError('fault_type参数错误, 请选择"bais_fault", "precision_degradation", "complete_failure", "draft"')
        return result

    def __verify_value(self, x):
        # 验证输入数据是否合法,若x为一维数组,则self.value必须为一个数值,若x为二维数组,则self.value可以为一个数值或者一个一维数组，但不能为二维数组
        if x.ndim == 1:
            if isinstance(self.fault_value, (int, float)):
                pass
            else:
                raise ValueError("x为一维数组时，self.value必须为一个数值")
        elif x.ndim == 2:
            if isinstance(self.fault_value, (int, float)):
                pass
            elif isinstance(self.fault_value, (list, np.ndarray)):
                if len(self.fault_value) != x.shape[1]:
                    raise ValueError("x为二维数组时，self.value必须为一个数值或者一个一维数组，但不能为二维数组")
            else:
                raise ValueError("x为二维数组时，self.value必须为一个数值或者一个一维数组，但不能为二维数组")


import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    normalData = pd.read_csv(r"..\数据\正常数据.txt", sep=';').values[:, 1:].astype(np.float64)
    sfg = SensorFaultGenerator(fault_type='draft', fault_value=[0.01, 100], dt=1)
    fault = sfg.fit_transform(normalData[:, 0:2])
    print(sfg.__dict__)
    plt.plot(normalData[:, 0])
    plt.plot(normalData[:, 1])

    plt.plot(fault)
    plt.show()

