import numpy as np
from 非参数概率密度估计 import ProbabilityDensityEstimation
from tqdm import tqdm


class MutualInformation:
    def __init__(self, method='histogram', bins=20, k=5, algorithm='auto', metric='minkowski', p=2, n_jobs=-1, kernel="gaussian", bandwidth=1.0, MinDouble=1e-15):
        """
        :param method:  概率密度估计方法，可选参数有：histogram、knn、kernel
        :param bins:  直方图的区间数
        :param k:  K近邻的K值
        :param algorithm:  近邻算法，可选参数有：auto、ball_tree、kd_tree、brute
        :param metric:  距离度量，可选参数有：euclidean、manhattan、chebyshev、minkowski、wminkowski、seuclidean、mahalanobis
        :param p:  距离度量的参数
        :param n_jobs:  并行数
        :param kernel:  核函数，可选参数有：gaussian、tophat、epanechnikov、exponential、linear、cosine
        :param bandwidth:  带宽
        :param MinDouble:  防止概率为0
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
        self.MinDouble = MinDouble

    def __union_probability(self, x, y):
        # 计算联合概率密度函数
        p_xy = ProbabilityDensityEstimation(method=self.method, kernel=self.kernel, bandwidth=self.bandwidth, bins=self.bins,
                                            k=self.k, algorithm=self.algorithm, metric=self.metric, p=self.p, n_jobs=self.n_jobs)
        p_xy.fit(np.hstack((x, y)))
        return p_xy

    def __marginal_probability(self, x, y):
        # 计算边缘概率密度函数
        p_x = ProbabilityDensityEstimation(method=self.method, kernel=self.kernel, bandwidth=self.bandwidth, bins=self.bins,
                                            k=self.k, algorithm=self.algorithm, metric=self.metric, p=self.p, n_jobs=self.n_jobs)
        p_x.fit(x)
        p_y = ProbabilityDensityEstimation(method=self.method, kernel=self.kernel, bandwidth=self.bandwidth, bins=self.bins,
                                            k=self.k, algorithm=self.algorithm, metric=self.metric, p=self.p, n_jobs=self.n_jobs)
        p_y.fit(y)
        return p_x, p_y

    def __mutual_information(self, x, y):
        # 计算互信息
        p_xy = self.__union_probability(x, y)
        p_x, p_y = self.__marginal_probability(x, y)
        # 使用多重积分计算互信息
        mi = 0
        h_x = 0
        h_y = 0
        # for i in tqdm(range(len(x)), desc='计算互信息', ncols=100, ascii=True):
        #     for j in range(len(y)):
        #         p_xy_prob = p_xy.predict(np.array([[x[i][0], y[j][0]]])) + self.MinDouble
        #         p_x_prob = p_x.predict(np.array([x[i]])) + self.MinDouble
        #         p_y_prob = p_y.predict(np.array([y[j]])) + self.MinDouble
        #         mi += p_xy_prob * np.log2(p_xy_prob / (p_x_prob * p_y_prob))  # 参考链接：https://www.omegaxyz.com/2018/08/02/mi/
        # for i in tqdm(range(len(x)), desc='计算x熵', ncols=100, ascii=True):
        #     p_x_prob = p_x.predict(np.array([x[i]])) + self.MinDouble
        #     h_x += -p_x_prob * np.log2(p_x_prob)
        # for j in tqdm(range(len(y)), desc='计算y熵', ncols=100, ascii=True):
        #     p_y_prob = p_y.predict(np.array([y[j]])) + self.MinDouble
        #     h_y += -p_y_prob * np.log2(p_y_prob)
        # x中的数与y中的数一一对应，所以可以使用矩阵运算
        # 使用 meshgrid 生成二维数组
        X, Y = np.meshgrid(x, y)
        # 将 X 和 Y 组合为一个二维数组
        combined = np.column_stack((X.ravel(), Y.ravel()))
        p_xy_prob = p_xy.predict(combined) + self.MinDouble
        p_x_prob = p_x.predict(np.array(x)) + self.MinDouble
        p_y_prob = p_y.predict(np.array(y)) + self.MinDouble
        for i in tqdm(range(len(x)), desc='计算互信息', ncols=100, ascii=True):
            for j in range(len(y)):
                p_xy_prob_temp = p_xy_prob[i*len(y)+j]
                p_x_prob_temp = p_x_prob[i]
                p_y_prob_temp = p_y_prob[j]
                mi += p_xy_prob_temp * np.log2(p_xy_prob_temp / (p_x_prob_temp * p_y_prob_temp))  # 参考链接：https://www.omegaxyz.com/2018/08/02/mi/
        # 计算x熵
        p_x_prob = p_x.predict(np.array(x)) + self.MinDouble
        h_x = np.sum(-p_x_prob * np.log2(p_x_prob))
        # 计算y熵
        p_y_prob = p_y.predict(np.array(y)) + self.MinDouble
        h_y = np.sum(-p_y_prob * np.log2(p_y_prob))
        # 归一化
        mi = 2*mi / (h_x + h_y)

        return mi

    def fit_transform(self, x, y):
        """
        :param x:  样本
        :param y:  样本
        :return:  None
        """
        x, y = np.array(x), np.array(y)
        # 检查x与y的合法性
        self.__check_validity(x, y)
        # 将x与y转换为（-1， 1）维数组
        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)
        # 计算互信息
        mi = self.__mutual_information(X, Y)
        return mi

    def __check_validity(self, x, y):
        # 检查x与y的维度是否相同
        if x.ndim != y.ndim:
            raise ValueError("x与y的维度不同")
        # 检查x与y的长度是否相同
        if len(x) != len(y):
            raise ValueError("x与y的长度不同")
        # 检查x与y的维度是否为1
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x与y的维度需要为1")

import matplotlib.pyplot as plt
if __name__ == '__main__':
    # 生成数据, x, y，其中x与y有较强的相关性
    x = np.arange(0, 1000, 1)
    y = x
    plt.scatter(x, y)
    plt.show()
    # 计算互信息
    mi = MutualInformation(method='kernel')
    result = mi.fit_transform(x, y)
    print("互信息为：%f" % (result))

    import numpy as np
    from sklearn.metrics import mutual_info_score

    # 计算互信息
    mi = mutual_info_score(x, y)

    print("互信息:", mi)