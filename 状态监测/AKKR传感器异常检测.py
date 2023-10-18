#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :AKKR传感器异常检测.py
# @Time      :2023/9/28 13:51
# @Author    :LongFei Shan
import numpy as np
import pandas as pd
import aakr
import matplotlib.pyplot as plt
from 传感器定位柱状图 import sensor_location_bar_chart
from SensorFaultDetect import SensorFaultDetect
# 画图中文显示
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


class AAKR(SensorFaultDetect):
    def __init__(self, metric='euclidean', bw=1., modified=False, penalty=None, n_jobs=-1):
        """

        :param metric:  距离度量
        :param bw:  带宽
        :param modified:  是否使用修改后的AKKR
        :param penalty:  惩罚项
        :param n_jobs:  并行数
        """
        super().__init__()
        self.metric = metric
        self.bw = bw
        self.modified = modified
        self.penalty = penalty
        self.n_jobs = n_jobs
        self.__akr = aakr.AAKR(metric=self.metric, bw=self.bw, modified=self.modified, penalty=self.penalty, n_jobs=self.n_jobs)

    def fit(self, x):
        """
        训练模型
        :param x:  正常数据
        :return:
        """
        self.__x = x
        self.__akr.fit(x)
    
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
    
    def predict(self, x, error_type='obsolute'):
        """
        预测
        :param x:  输入数据
        :param error_type:  误差类型, 'obsolute'为绝对误差, 'relative'为相对误差, 'euclidean'为欧式距离
        :return:
        """
        result = self.__akr.transform(x)
        # 计算输入数据与结果之间的差值与差值之和的比值,作为异常指标
        error = None
        if error_type == 'obsolute':
            error = np.abs(x - result)
        elif error_type == 'relative':
            error = np.abs(x - result) / (np.abs(x) + 1e-11)
        elif error_type == 'euclidean':
            error = np.sqrt(np.sum(np.square(x - result), axis=1, keepdims=True))
        else:
            raise ValueError('error_type参数错误, 请选择"obsolute"或者"relative"')
        result = np.sum(error, axis=1, keepdims=True)
        return result
    
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

    def sensor_fault_location_predict(self, X, error_type='obsolute'):
        """
        传感器异常检测
        :param X:  输入数据
        :param error_type:  误差类型, 'obsolute'为绝对误差, 'relative'为相对误差
        :return:
        """
        result = self.__akr.transform(X)
        # 计算输入数据与结果之间的差值与差值之和的比值,作为异常指标
        error = None
        if error_type == 'obsolute':
            error = np.abs(X - result)
        elif error_type == 'relative':
            error = np.abs(X - result) / (np.abs(X) + 1e-11)
        else:
            raise ValueError('error_type参数错误, 请选择"obsolute"或者"relative"')
        ratio = error / np.sum(error, axis=1, keepdims=True)
        return ratio


if __name__ == "__main__":
    normalData = pd.read_csv(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\正常数据.txt", sep=';').values[:, 1:].astype(np.float64)
    fault = pd.read_csv(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\一环路冷却剂流量-0.001-噪声波动.txt", sep=',').values[:, 1:].astype(np.float64)
    normal_testData = pd.read_excel(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\100%normal.xlsx").values[:, 1:].astype(np.float64)
    akr = AAKR(metric='euclidean', bw=1., modified=False, penalty=None, n_jobs=-1)
    akr.fit(normalData)
    akr.save_model(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\结果\AAKR\AAKR.pkl')
    akr = AAKR.load_model(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\结果\AAKR\AAKR.pkl')
    # 异常检测
    result = akr.predict(fault, error_type='obsolute')
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
        sensor_location_bar_chart(result[i, :].reshape(1, -1), up_threshold, down_threshold, title=f'AKKR-故障传感器-{i}', xlabel='传感器编号', ylabel='传感器数量')
        plt.savefig(f'D:/文件/小论文/基于D-S证据理论的传感器故障定位方法研究/代码/结果/AAKR/AKKR-故障传感器-{i}.png')
        plt.close()


