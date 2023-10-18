import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from 直方图概率密度估计 import Histogram


class KernelDensityEstimation:
    def __init__(self, kernel="gaussian", bandwidth=1.0):
        """
        :param kernel: 核函数，可选参数有：gaussian、tophat、epanechnikov、exponential、linear、cosine
        :param bandwidth: 带宽
        """
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit(self, x):
        """
        :param x: 样本
        :return: None
        """
        self.x = x
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kde.fit(self.x)

    def predict(self, x):
        """
        :param x: 样本
        :return: probability_density, probability_density为概率密度估计值
        """
        # 计算概率密度估计值
        probability_density = np.exp(self.kde.score_samples(x))
        return probability_density
