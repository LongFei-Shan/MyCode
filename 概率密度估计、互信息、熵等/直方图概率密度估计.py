import numpy as np
import matplotlib.pyplot as plt


class Histogram:
    def __init__(self, bins=20):
        self.bins = bins

    def fit(self, x):
        self.__x = np.array(x)
        self.hist, self.bins = np.histogramdd(x, bins=self.bins, density=True)
    
    def predict(self, X):
        # 1、找到每个样本所属的区间
        probability_density = []  # 每个样本所属区间的概率密度
        probability = []  # 每个样本所属区间的概率
        # 检查是否调用fit函数
        if not hasattr(self, 'hist'):
            raise ValueError('请先调用fit函数')
        self.__check_validity(X)  # 检查X是否合法
        # 由于np.histogramdd输出为二维，若X为一维，那么需要转换为一维
        if X.ndim == 1:
            self.bins = self.bins[0]
        # 预测概率与概率密度
        for x in X:
            # 判断是否为多维数据
            if X.ndim >= 2:
                index = self.__find_interval_dd(x, self.bins)
                # 2、计算每个样本所属区间的概率密度
                probability_density_temp, probability_temp = self.__find_probaility_dd(index)
                probability_density.append(probability_density_temp)
                probability.append(probability_temp)
            else:
                index = self.__find_interval(x, self.bins)
                # 2、计算每个样本所属区间的概率密度
                probability_density_temp, probability_temp = self.__find_probaility(index)    
                probability_density.append(probability_density_temp)
                probability.append(probability_temp)
            
        return np.array(probability_density)
    
    def __check_validity(self, X):
        # 先判断X与self.__x的维度是否相同
        if X.ndim != self.__x.ndim:
            raise ValueError('X与训练数据x的维度不同')
        if X.ndim >= 2:
            # 判断X与self.__x的其他维度是否相同
            for i in range(1, X.ndim):
                if X.shape[i] != self.__x.shape[i]:
                    raise ValueError('X与self.__x的维度不同')
    
    def __find_probaility_dd(self, index):
        # 1、先判断每一个维度的index是否为0，如果为0，则该维度的概率为0，或者是否为最后一个区间，如果是，则该维度的概率为0
        for i in range(len(index)):
            if index[i] <= 0 or index[i] >= len(self.bins[i]) - 1:
                return 0, 0
        # 2、如果不是，则计算该维度的概率
        probability_density_temp = self.hist[index[0]]
        for i in range(1, len(index)):
            probability_density_temp = probability_density_temp[i]
        probability_density = probability_density_temp
        # 3、计算面积
        probability = probability_density
        for i in range(len(index)):
            probability *= (self.bins[i][index[i]] - self.bins[i][index[i]-1])
        return probability_density, probability
    
    def __find_probaility(self, index):
        # 1、先判断index是否为0，如果为0，则概率为0，或者是否为最后一个区间，如果是，则概率为0
        if index <= 0 or index >= len(self.bins) - 1:
            return 0, 0
        # 2、如果不是，则计算概率
        else:
            probability_density = self.hist[index]
            probability = self.hist[index] * (self.bins[index] - self.bins[index-1])
        return probability_density, probability
    
    def __find_interval_dd(self, x, intervals):
        """
        定位多维直方图中样本所属的区间
        :param x: 样本
        :param intervals: 区间
        :return: index, index为x所属区间的索引
        """
        index = []
        for i in range(len(x)):
            index.append(self.__find_interval(x[i], intervals[i]))
        return index

    def __find_interval(self, x, intervals):
        """
        定位一维直方图中样本所属的区间
        :param x: 样本
        :param intervals: 区间
        :return: index, index为x所属区间的索引, 样本x的区间为[index-1,index]
        """
        # 使用二分法查找
        low = 0
        high = len(intervals) - 1
        # 判断x是否大于intervals最大值或者小于最小值
        if x >= intervals[-1]:
            return len(intervals) - 1
        if x <= intervals[0]:
            return 0

        while low <= high:
            mid = (low + high) // 2
            if x < intervals[mid]:
                if mid == 0 or x > intervals[mid-1]:
                    return mid
                high = mid
            elif x > intervals[mid]:
                if mid == len(intervals) - 1 or x < intervals[mid+1]:
                    return mid + 1
                low = mid
            else:
                return mid

        return low
