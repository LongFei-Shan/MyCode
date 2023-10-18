#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MSET.py
# @Time      :2023/9/28 16:20
# @Author    :LongFei Shan
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from 传感器定位柱状图 import sensor_location_bar_chart
from sklearn.preprocessing import MinMaxScaler
from SensorFaultDetect import SensorFaultDetect
# 画图中文显示
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


class MSET(SensorFaultDetect):
    def __init__(self, name='euclidean', error_type='obsolute'):
        """
        :param name:  距离度量名称 'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski', 'correlation'
        :param error_type:  误差类型, 'obsolute'为绝对误差, 'relative'为相对误差
        """
        self.name = name
        self.error_type = error_type

    def fit(self, x):
        """
        训练模型
        :param x:  正常数据, shape=(n_samples, n_features)
        :return:
        """
        self.__x = x
        self.mms = MinMaxScaler()
        x = self.mms.fit_transform(x)
        x = np.array(x)
        self.__D = x.T

    def __operator_nolinear(self, x, y, name="linear"):
        """
        非线性运算符
        :param x:
        :param y:
        :return:
        """
        # 构建一个x.shape[0]行, y.shape[1]列的矩阵
        result = np.zeros((x.shape[0], y.shape[1]))
        if name == "linear":
            return np.dot(x, y)

        for i in range(x.shape[0]):
            for j in range(y.shape[1]):
                if name == "euclidean":
                    result[i, j] = distance.euclidean(x[i, :], y[:, j])
                elif name == "manhattan":
                    result[i, j] = distance.cityblock(x[i, :], y[:, j])
                elif name == "cosine":
                    result[i, j] = distance.cosine(x[i, :], y[:, j])
                elif name == "chebyshev":
                    result[i, j] = distance.chebyshev(x[i, :], y[:, j])
                elif name == "minkowski":
                    result[i, j] = distance.minkowski(x[i, :], y[:, j])
                elif name == "correlation":
                    result[i, j] = distance.correlation(x[i, :], y[:, j])
                else:
                    raise ValueError("运算符名称错误")
        return result

    def __verify_value(self, x):
        # 验证x与self.__D的维度是否一致
        if x.ndim != self.__D.ndim :
            raise ValueError("x与self.__D的维度不一致")


    def __predict(self, x):
        """
        预测
        :param x:  测试数据
        :return:
        """
        x = np.array(x)
        # 归一化
        x = self.mms.transform(x)
        # 转置
        x = x.T
        # 验证x与self.__D的维度是否一致
        self.__verify_value(x)
        # 计算权重矩阵
        self.__W_left = np.linalg.inv(self.__operator_nolinear(self.__D.T, self.__D, name=self.name))
        self.__W_right = self.__operator_nolinear(self.__D.T, x, name="linear")
        self.__W = np.dot(self.__W_left, self.__W_right)
        # 计算测试数据的估计值
        x_est = np.dot(self.__D, self.__W)
        # 计算输入数据与结果之间的差值与差值之和的比值,作为异常指标
        error = None
        if self.error_type == 'obsolute':
            error = np.abs(x - x_est)
        elif self.error_type == 'relative':
            error = np.abs(x - x_est) / (np.abs(x) + 1e-11)
        else:
            raise ValueError('error_type参数错误, 请选择"obsolute"或者"relative"')

        return error
    
    def predict(self, x):
        """anomaly detection model prediction

        Args:
            x (np.ndarray): test data

        Returns:
            np.ndarray: anomaly index
        """
        error = self.__predict(x)
        error = error.T
        result = np.sum(error, axis=1, keepdims=True)

        return result
    
    def threshold(self, x=None):
        """anomaly detection model threshold

        Args:
            x (np.ndarray, optional): normal data. Defaults to None.

        Returns:
            tuple: up_threshold, down_threshold, up_threshold为上限阈值, down_threshold为下限阈值
        """
        if x is None:
            result = self.predict(self.__x)
            return self._box_plot(result, is_zero=False)
        else:
            result = self.predict(x)
            return self._box_plot(result, is_zero=False)
        
    
    def sensor_fault_location_threshold(self, x=None):
        """sensor fault location model threshold

        Args:
            x (_type_, optional): normal data. Defaults to None.

        Returns:
            tuple: up_threshold, down_threshold, up_threshold为上限阈值, down_threshold为下限阈值
        """
        if x is None:
            result = self.sensor_fault_location_predict(self.__x)
            return self._box_plot(result, is_zero=True)
        else:
            result = self.sensor_fault_location_predict(x)
            return self._box_plot(result, is_zero=True)

    def sensor_fault_location_predict(self, X):
        """
        传感器异常检测
        :param X:  输入数据
        :return: 传感器异常指标
        """
        error = self.__predict(X)
        error = error.T
        ratio = error / np.sum(error, axis=1, keepdims=True)
        return ratio


if __name__ == "__main__":
    normalData = pd.read_csv(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\正常数据.txt", sep=';').values[:, 1:].astype(np.float64)
    fault = pd.read_csv(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\一环路冷却剂流量-0.001-噪声波动.txt", sep=',').values[:, 1:].astype(np.float64)
    normal_testData = pd.read_excel(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\100%normal.xlsx").values[:, 1:].astype(np.float64)
    akr = MSET(name='euclidean')
    akr.fit(normalData)
    akr.save_model(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\结果\AAKR\AAKR.pkl')
    akr = MSET.load_model(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\结果\AAKR\AAKR.pkl')
    # 异常检测
    result = akr.predict(fault)
    up_threshold, down_threshold = akr.threshold(normal_testData)
    plt.plot(result)
    plt.axhline(up_threshold, color='r', linestyle='--')
    plt.axhline(down_threshold, color='r', linestyle='--')
    plt.title('AKKR-异常检测')
    plt.xlabel('数据编号')
    plt.ylabel('异常指标')
    plt.show()
    
    # 计算阈值
    up_threshold, down_threshold = akr.sensor_fault_location_threshold(normal_testData)
    result = akr.sensor_fault_location_predict(fault)
    # 画出每一行柱状图
    # 计算每一个传感器的异常指标是否超过阈值，若超出阈值则认为该传感器异常, color为异常传感器的颜色red, 否则为green
    for i in range(result.shape[0]):
        # 画出每一行柱状图
        sensor_location_bar_chart(result[i, :].reshape(1, -1), up_threshold, down_threshold, title=f'MEST-故障传感器-{i}', xlabel='传感器编号', ylabel='传感器数量')
        plt.savefig(f'D:/文件/小论文/基于D-S证据理论的传感器故障定位方法研究/代码/结果/MEST/MEST-故障传感器-{i}.png')
        plt.close()
