#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :传感器定位柱状图.py
# @Time      :2023/9/28 16:13
# @Author    :LongFei Shan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sensor_location_bar_chart(result, up_threshold, down_threshold, title='传感器定位柱状图', xlabel='传感器编号', ylabel='传感器数量'):
    """
    传感器定位柱状图
    :param data:  传感器定位数据
    :param title:  图片标题
    :param xlabel:  x轴标签
    :param ylabel:  y轴标签
    :param save_path:  保存路径
    :return:
    """
    color = []
    for j in range(result.shape[1]):
        if result[0, j] > up_threshold[j] or result[0, j] < down_threshold[j]:
            color.append('red')
        else:
            color.append('green')
    # 画出每一行柱状图
    plt.bar(range(result.shape[1]), result[0, :], color=color)
    plt.bar(range(result.shape[1]), up_threshold, color='blue', bottom=down_threshold, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


if __name__ == "__main__":
    pass
