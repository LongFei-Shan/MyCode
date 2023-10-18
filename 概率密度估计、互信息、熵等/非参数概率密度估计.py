import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from 直方图概率密度估计 import Histogram
from K近邻概率密度估计 import KNNEstimator
from 核函数概率密度估计 import KernelDensityEstimation


class ProbabilityDensityEstimation:
    def __init__(self, method='histogram', bins=20, k=5, algorithm='auto', metric='minkowski', p=2, n_jobs=-1,
                 kernel="gaussian", bandwidth=1.0):
        """
        :param method: 概率密度估计方法，可选参数有：histogram、knn、kernel
        :param bins: 直方图的区间数
        :param k: K近邻的K值
        :param algorithm: 近邻算法，可选参数有：auto、ball_tree、kd_tree、brute
        :param metric: 距离度量，可选参数有：euclidean、manhattan、chebyshev、minkowski、wminkowski、seuclidean、mahalanobis
        :param p: 距离度量的参数
        :param n_jobs: 并行数
        :param kernel: 核函数，可选参数有：gaussian、tophat、epanechnikov、exponential、linear、cosine
        :param bandwidth: 带宽
        """
        self.method = method
        self.bins = bins
        self.k = k
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit(self, x):
        """
        :param x: 样本
        :return: None
        """
        self.x = x
        if self.method == 'histogram':
            self.hist = Histogram(bins=self.bins)
            self.hist.fit(self.x)
        elif self.method == 'knn':
            self.knn = KNNEstimator(k=self.k, algorithm=self.algorithm, metric=self.metric, p=self.p,
                                    n_jobs=self.n_jobs)
            self.knn.fit(self.x)
        elif self.method == 'kernel':
            self.kde = KernelDensityEstimation(kernel=self.kernel, bandwidth=self.bandwidth)
            self.kde.fit(self.x)
        else:
            raise ValueError('method参数错误')

    def predict(self, x):
        """
        :param x: 样本
        :return: probability_density, probability_density为概率密度
        """
        # 判断x是否为一维数组，如果是，则报错
        if x.ndim == 1:
            raise ValueError("x的维度需要为>1数组，row为样本数，col为特征数")
        if self.method == 'histogram':
            probability_density = self.hist.predict(x)
        elif self.method == 'knn':
            probability_density = self.knn.predict(x)
        elif self.method == 'kernel':
            probability_density = self.kde.predict(x)
        else:
            raise ValueError('method参数错误')
        return probability_density


if __name__ == '__main__':
    # 生成数据
    np.random.seed(0)
    x = np.concatenate((np.random.normal(0, 1, (1000, 1)), np.random.normal(5, 1, (1000, 1))))
    # 生成测试数据
    x_test = np.linspace(-5, 10, 1000)
    x_test = x_test.reshape(-1, 1)
    # 概率密度估计
    pde = ProbabilityDensityEstimation(method='knn', bins=20)
    pde.fit(x)
    probability_density = pde.predict(x_test)
    # 画图
    plt.plot(x_test, probability_density)
    plt.show()