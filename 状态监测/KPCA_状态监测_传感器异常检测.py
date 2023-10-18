import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from scipy.stats import f
from scipy import stats
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import KernelPCA
# 中文以及负号失效问题
plt.rcParams['font.sans-serif'] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


class KernelPrincipalComponentAnalysis:
    def __init__(self, n_components=2, n_rate=0.85, kernel="linear", gamma=1, degree=3, coef0=0, tol=1e-4, max_iter=1000, is_normalization=True, is_rate=True):
        """
        Parameters
        ----------
        n_components : int, default=2, 降维后的维度
        kernel : str, default="linear", 核函数，可选：linear, poly, rbf, sigmoid, gaussian
        """
        self.n_components = n_components
        self.n_rate = n_rate
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.is_normalization = is_normalization
        self.is_rate = is_rate

    def __normalization(self, X):
        """
        数据归一化
        :param X: 数据
        :return: 归一化后的数据
        """
        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(X)
        return X

    def gaussianKernelFunction(self, x, y, gamma):
        """
        高斯核函数
        -------------
        :param x:
        :param y:
        :param gamma:
        :return:
        """
        result = np.exp(-gamma * np.linalg.norm(x - y, ord=2))
        return result

    def linearKernelFunction(self, x, y, coef0=0):
        """
        线性核函数
        -------------
        :param x:
        :param y:
        :return:
        """
        result = np.dot(x, y) + coef0
        return result

    def polyKernelFunction(self, x, y, gamma=1, degree=3, coef0=0):
        """
        多项式核函数
        -------------
        :param x:
        :param y:
        :param gamma:
        :param degree:
        :param coef0:
        :return:
        """
        result = (gamma * np.dot(x, y) + coef0) ** degree
        return result

    def sigmoidKernelFunction(self, x, y, gamma=1, coef0=0):
        """
        sigmoid核函数
        -------------
        :param x:
        :param y:
        :param gamma:
        :param coef0:
        :return:
        """
        result = np.tanh(gamma * np.dot(x, y) + coef0)
        return result

    def kernelFunction(self, x, y, kernel="linear", gamma=1, degree=3, coef0=0):
        """
        核函数
        -------------
        :param x:
        :param y:
        :param kernel:
        :param gamma:
        :param degree:
        :param coef0:
        :return:
        """
        if kernel == "linear":
            result = self.linearKernelFunction(x, y, coef0)
        elif kernel == "poly":
            result = self.polyKernelFunction(x, y, gamma, degree, coef0)
        elif kernel == "rbf" or kernel == "gaussian":
            result = self.gaussianKernelFunction(x, y, gamma)
        elif kernel == "sigmoid":
            result = self.sigmoidKernelFunction(x, y, gamma, coef0)
        else:
            raise ValueError("核函数类型错误")
        return result

    def centerKernelMatrix(self, K):
        """
        中心化核矩阵
        -------------
        :param K:
        :return:
        """
        K = K - self.mean_K
        return K

    def __contribution_rate(self, s):
        """
        计算贡献率
        :param s: 奇异值
        :return: 贡献率
        """
        s_sum = np.sum(s)
        s_rate = s / s_sum
        return s_rate

    def __get_n_component(self, s_rate):
        """
        获取主成分个数
        :param s_rate: 贡献率
        :return: 主成分个数
        """
        for i in range(len(s_rate)):
            if np.sum(s_rate[:i+1]) >= self.n_rate:
                return i+1

    def SquaredPredictionError(self, x):
        """
        计算预测误差SPE
        :param X: 数据
        :return: 预测误差
        """
        K = self.calcuate_K(x, self.X)
        self.residual = K@self.eig_vectors[:, self.n_components:]@self.eig_vectors[:, self.n_components:].T
        SPE = np.sum(np.abs(self.residual), axis=1)  # 按行求平均值,得到每个样本的预测误差SPE
        return SPE

    def SPEThreshold(self):
        """
        计算SPE阈值
        :return:
        """
        theta1 = np.sum(self.eig_values[self.n_components:])
        theta2 = np.sum(self.eig_values[self.n_components:]**2)
        theta3 = np.sum(self.eig_values[self.n_components:]**3)
        h0 = 1 - 2*theta1*theta3/(3*theta2**2)
        c = stats.norm.ppf(1-0.05)
        SPE_threshold = theta1*(1+c*h0*np.sqrt(2*theta2)/theta1 + theta2*h0*(h0-1)/theta1**2)**(1/h0)
        return SPE_threshold

    def SPE_FeatureRate(self, x):
        """
        计算SPE特征率
        :param x: 数据
        :return:
        """
        K = self.calcuate_K(x, self.X)
        residual_K = K@self.eig_vectors[:, self.n_components:]
        residual = self.inverse_transform(residual_K)
        SPE_FeatureRate = np.abs(residual)/np.sum(np.abs(residual), axis=1, keepdims=True)
        return SPE_FeatureRate

    def SPE_FeatureRateThreshold(self, x):
        """
        SPE贡献率阈值
        :param x: 数据
        :return:DownLimit, UpLimit
        """
        SPE_FeatureRate = self.SPE_FeatureRate(x)
        UpLimit = []
        DownLimit = []
        for i in range(SPE_FeatureRate.shape[1]):
            temp_downlimit, temp_uplimit = self.BoxModel(SPE_FeatureRate[:, i])
            UpLimit.append(temp_uplimit)
            DownLimit.append(temp_downlimit)
        return DownLimit, UpLimit

    def HotellingT2Threshold(self):
        """
        计算Hotelling T2统计量阈值
        :return:
        """
        k = self.n_components
        n = self.eig_vectors.shape[0]
        alpha = 0.05
        F = f.ppf(1 - alpha, k, n-k)
        # 计算阈值
        T2_threshold =((k*(n-1))/((n-k)))*F
        return T2_threshold

    def HotellingT2(self, x):
        """
        计算Hotelling T2统计量
        :param X: 数据
        :return: Hotelling T2统计量
        """
        K = self.calcuate_K(x, self.X)
        T2 = np.sum((K@self.eig_vectors[:, :self.n_components]@np.diag(np.sqrt(1/np.abs(self.eig_values[:self.n_components]))))**2, axis=1)
        return T2

    def HotellingT2_FeatureRate(self, x):
        """
        计算Hotelling T2统计量特征率
        :param x: 数据
        :return:
        """
        K = self.calcuate_K(x, self.X)
        T = []
        for X in K:
            t = self.eig_vectors@X.T
            T_i = []
            for i in range(self.eig_values.shape[0]):
                x_i = X[i]
                p_i = self.eig_vectors[i, :]
                temp = 0
                for j in range(self.eig_values.shape[0]):
                    temp += t[j]*p_i[j]*x_i/self.eig_values[j]**2
                T_i.append(temp)
            T.append(T_i)
        T2_FeatureRate = np.array(T)/np.mean(T, axis=1, keepdims=True)
        return T2_FeatureRate

    def T2_FeatureRateThreshold(self, x):
        """
        T2贡献率阈值
        :param x: 数据
        :return:
        """
        T2_FeatureRate = self.HotellingT2_FeatureRate(x)
        UpLimit = []
        DownLimit = []
        for i in range(x.shape[1]):
            temp_downlimit, temp_uplimit = self.BoxModel(T2_FeatureRate[:, i])
            UpLimit.append(temp_uplimit)
            DownLimit.append(temp_downlimit)
        return DownLimit, UpLimit

    def BoxModel(self, data):
        """
        箱型图
        :param data: 正常数据
        :return: DownLimit, UpLimit，上下限阈值
        """
        # 箱型图
        temp = np.sort(data)
        # 下四分位数
        Q1 = temp[int(len(data) / 4)]
        # 上四分位数
        Q3 = temp[-int(len(data) / 4)]
        # 四分位距离
        IQR = Q3 - Q1
        # 上限
        UpLimit = Q3 + 1.5 * IQR
        # 下限
        DownLimit = Q1 - 1.5 * IQR

        return DownLimit, UpLimit

    def calcuate_K(self, x, y=None):
        if self.is_normalization:
            if x.ndim == 1:
                raise ValueError("x must is 2 dim!")
            if not (y is None):
                x = self.scaler.transform(x)
                y = self.scaler.transform(y)
            else:
                x = self.__normalization(x)
        n = x.shape[0]
        K = np.zeros((n, n))
        m = n
        if not (y is None):
            n = y.shape[0]
            m = x.shape[0]
            K = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if not (y is None):
                    K[i, j] = self.kernelFunction(x[i, :], y[j, :], self.kernel, self.gamma, self.degree, self.coef0)
                else:
                    K[i, j] = self.kernelFunction(x[i, :], x[j, :], self.kernel, self.gamma, self.degree, self.coef0)
        if y is None:
            self.mean_K = np.mean(K, axis=0, keepdims=True)
        K = self.centerKernelMatrix(K)
        return K

    def fit(self, X):
        """
        训练模型
        -------------
        :param X:
        :return:
        """
        # 数据归一化
        self.X = np.array(X, dtype=np.float64)
        K = self.calcuate_K(X)
        # 求解特征值和特征向量
        self.eig_values, self.eig_vectors = np.linalg.eig(K)
        # 去除虚部
        self.eig_values = np.real(self.eig_values)
        self.eig_vectors = np.real(self.eig_vectors)
        # 对特征值进行排序
        self.eig_values = self.eig_values[np.argsort(-self.eig_values)]
        self.eig_vectors = self.eig_vectors[:, np.argsort(-self.eig_values)]
        # 选取前n_component个特征值与特征向量
        s_rate = self.__contribution_rate(self.eig_values)
        if self.is_rate:
            self.n_components = self.__get_n_component(s_rate)
        self.dual_coef_ = linalg.inv(K)@self.scaler.transform(self.X)

    def predict(self, x):
        # 1、计算x与X之间的中心矩阵K
        K = self.calcuate_K(x, self.X)
        eig_values = self.eig_values[:self.n_components]
        eig_vectors = self.eig_vectors[:, :self.n_components]
        # 2、计算x的降维结果
        x = np.dot(K, eig_vectors)
        return x

    def inverse_transform(self, x):
        # x为使用kpca降维后的数据，将其还原成降维之前的数据
        # 1、将x还原成K
        K = x @ self.eig_vectors[:, self.n_components:].T
        # 2、将K还原成原始数据
        X = K@self.dual_coef_
        return X