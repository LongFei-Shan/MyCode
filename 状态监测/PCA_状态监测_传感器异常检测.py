import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from scipy.stats import f
from scipy import stats
import matplotlib.pyplot as plt
from SensorFaultDetect import SensorFaultDetect
from 传感器定位柱状图 import sensor_location_bar_chart
# 中文以及负号失效问题
plt.rcParams['font.sans-serif'] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


class PrincipalComponentAnalsys(SensorFaultDetect):
    def __init__(self, n_component=3, contribution_rate=0.85, is_rate=False, is_normalization=True, 
                 reconstruction_error=1e-11, normalization_type='StandardScaler', add_white_noise=False, std=0.1):
        """
        Parameters
        ----------
        :param n_component: 降维后的维度
        :param contribution_rate: 贡献率
        :param is_rate: 是否使用贡献率
        :param is_normalization: 是否归一化
        :param reconstruction_error: 重构误差
        :param normalization_type: 归一化类型, 'MinMaxScaler', 'StandardScaler'
        :param add_white_noise: 是否添加白噪声
        """
        self.n_component = n_component
        self.contribution_rate = contribution_rate
        self.is_rate = is_rate
        self.is_normalization = is_normalization
        self.reconstruction_error = reconstruction_error
        self.normalization_type = normalization_type
        self.add_white_noise = add_white_noise
        self.std = std

    def __matrix_decomposition(self, C):
        """
        矩阵分解
        :param C: 协方差矩阵
        :return:
        """
        u, s, vh = np.linalg.svd(C)
        return s, vh

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
            if np.sum(s_rate[:i+1]) >= self.contribution_rate:
                return i+1

    def __normalization(self, X):
        """
        数据归一化
        :param X: 数据
        :return: 归一化后的数据
        """
        if self.normalization_type == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif self.normalization_type == 'StandardScaler':
            self.scaler = StandardScaler()
        else:
            raise ValueError('normalization_type参数错误, 请选择"MinMaxScaler"或者"StandardScaler"')
        X = self.scaler.fit_transform(X)
        if self.add_white_noise:
            X += np.random.normal(0, self.std, size=X.shape)
        return X

    def __SquaredPredictionError(self, x):
        """
        计算预测误差SPE
        :param X: 数据
        :return: 预测误差
        """
        if self.is_normalization:
            x = self.scaler.transform(x)
        self.residual = np.dot(self.eigenvector[self.n_component:, :].T, np.dot(self.eigenvector[self.n_component:, :], x.T)).T
        SPE = np.mean(np.abs(self.residual), axis=1)  # 按行求平均值,得到每个样本的预测误差SPE
        return SPE

    def __SPEThreshold(self):
        """
        计算SPE阈值
        :return:
        """
        theta1 = np.sum(self.eigenvalue[self.n_component:])
        theta2 = np.sum(self.eigenvalue[self.n_component:]**2)
        theta3 = np.sum(self.eigenvalue[self.n_component:]**3)
        h0 = 1 - 2*theta1*theta3/(3*theta2**2)
        c = stats.norm.ppf(1-0.05)
        SPE_threshold = theta1*(1+c*h0*np.sqrt(2*theta2)/theta1 + theta2*h0*(h0-1)/theta1**2)**(1/h0)
        return SPE_threshold

    def __SPE_FeatureRate(self, x):
        """
        计算SPE特征率
        :param x: 数据
        :return:
        """
        if self.is_normalization:
            x = self.scaler.transform(x)
        residual = np.dot(self.eigenvector[self.n_component:, :].T, np.dot(self.eigenvector[self.n_component:, :], x.T)).T
        SPE_FeatureRate = np.abs(residual)**2/np.sum(np.abs(residual)**2, axis=1, keepdims=True)
        return SPE_FeatureRate

    def __SPE_FeatureRateThreshold(self, x):
        """
        SPE贡献率阈值
        :param x: 数据
        :return:DownLimit, UpLimit
        """
        SPE_FeatureRate = self.__SPE_FeatureRate(x)
        up_threshold, down_threshold = self._box_plot(SPE_FeatureRate, is_zero=True)
        return down_threshold, up_threshold

    def __HotellingT2Threshold(self):
        """
        计算Hotelling T2统计量阈值
        :return:
        """
        k = self.n_component
        n = self.eigenvalue.shape[0]
        alpha = 0.05
        F = f.ppf(1 - alpha, k, n-k)
        # 计算阈值
        T2_threshold =((k*(n**2-1))/(n*(n-k)))*F
        return T2_threshold

    def __HotellingT2(self, x):
        """
        计算Hotelling T2统计量
        :param X: 数据
        :return: Hotelling T2统计量
        """
        if self.is_normalization:
            x = self.scaler.transform(x)
        T2 = np.mean((x@self.eigenvector[:self.n_component, :].T@np.diag(np.sqrt(1/self.eigenvalue[:self.n_component])))**2, axis=1)
        return T2

    def __HotellingT2_FeatureRate(self, x):
        """
        计算Hotelling T2统计量特征率
        :param x: 数据
        :return:
        """
        if self.is_normalization:
            x = self.scaler.transform(x)
        T = []
        for X in x:
            t = self.eigenvector@X.T
            T_i = []
            for i in range(self.eigenvalue.shape[0]):
                x_i = X[i]
                p_i = self.eigenvector[i, :]
                temp = 0
                for j in range(self.eigenvalue.shape[0]):
                    temp += t[j]*p_i[j]*x_i/self.eigenvalue[j]**2
                T_i.append(temp)
            T.append(T_i)
        T = np.abs(T)
        T2_mean = np.sum(np.abs(T), axis=1, keepdims=True)
        T2_FeatureRate = np.abs(T)/T2_mean
        return T2_FeatureRate

    def __T2_FeatureRateThreshold(self, x):
        """
        T2贡献率阈值
        :param x: 数据
        :return:
        """
        T2_FeatureRate = self.__HotellingT2_FeatureRate(x)
        up_threshold, down_threshold = self._box_plot(T2_FeatureRate, is_zero=True)
        return down_threshold, up_threshold

    def SensorIterationReconstruction(self, x, index):
        """
        传感器数据迭代重构
        :param x:数据
        :param index:故障传感器索引
        :return:最佳估计值
        """
        if self.is_normalization:
            x = self.scaler.transform(x)
        C = self.eigenvector[:self.n_component, :].T@self.eigenvector[:self.n_component, :]
        if C[index, index] == 1:
            print("该传感器无法迭代重构！")
            return x@C[:, index]
        else:
            result = x @ C
            C1 = C[:, index]
            C1[index] = 0
            C2 = C[index, index]
            x_new = result[:, index]
            x_old = x[:, index]
            loop = 0
            while True:
                error = np.sum(np.abs(x_new - x_old))
                if error < self.reconstruction_error:
                    break
                x_old = x_new
                x_new = x@C1.reshape(-1, 1) + x_new*C2
                loop += 1
            result[:, index] = x_new
            print(f"重构迭代次数：{loop}, 重构误差：{error}")
        if self.is_normalization:
            result = self.scaler.inverse_transform(result)
        return result

    def fit(self, X):
        """
        训练模型
        :param X: 训练数据
        :return:
        """
        self.__x = X
        if self.is_normalization:
            X = self.__normalization(X)
        C = np.cov(X.T)  # 默认计算方式是以行进行
        self.eigenvalue, self.eigenvector = self.__matrix_decomposition(C)
        s_rate = self.__contribution_rate(self.eigenvalue)
        if self.is_rate:
            self.n_component = self.__get_n_component(s_rate)
        self.P = self.eigenvector[:self.n_component, :]
        self.X_new = np.dot(self.P, X.T).T

    def transform(self, x):
        """
        转换数据
        :param X: 转换数据
        :return: 转换后的数据
        """
        if self.is_normalization:
            x = self.scaler.transform(x)
        return np.dot(self.P, x.T).T

    def threshold(self):
        """
        异常检测-训练

        :return: SPE_threshold（SPE统计量阈值）, T2_threshold（T2统计量阈值）
        """
        SPE_threshold = self.__SPEThreshold()
        T2_threshold = self.__HotellingT2Threshold()
        return SPE_threshold, T2_threshold

    def predict(self, x):
        """
        异常检测-预测

        :param x: 输入数据
        :return: SPE(SPE统计量)，T2(T2统计量)
        """
        T2 = self.__HotellingT2(x)
        SPE = self.__SquaredPredictionError(x)
        return SPE, T2

    def sensor_fault_location_threshold(self):
        """
        传感器故障定位-训练-获取阈值
        
        :return: SPE_FeatureRate_DownLimit（SPE统计量阈值下限-传感器定位）,
                 SPE_FeatureRate_UpLimit（SPE统计量阈值上限-传感器定位）,
                 T2_FeatureRate_DownLimit（T2统计量阈值下限-传感器定位）,
                 T2_FeatureRate_UpLimit（T2统计量阈值上限-传感器定位）
        """
        T2_FeatureRate_DownLimit, T2_FeatureRate_UpLimit = self.__T2_FeatureRateThreshold(self.__x)
        SPE_FeatureRate_DownLimit, SPE_FeatureRate_UpLimit = self.__SPE_FeatureRateThreshold(self.__x)
        return SPE_FeatureRate_DownLimit, SPE_FeatureRate_UpLimit, T2_FeatureRate_DownLimit, T2_FeatureRate_UpLimit

    def sensor_fault_location_predict(self, x):
        """
        传感器故障定位-预测

        :param x: 输入数据
        :return: SPE_FeatureRate（每个传感器SPE特征贡献率）, T2_FeatureRate（每个传感器T2特征贡献率）
        """
        T2_FeatureRate = self.__HotellingT2_FeatureRate(x)
        SPE_FeatureRate = self.__SPE_FeatureRate(x)
        return SPE_FeatureRate, T2_FeatureRate

    def inverse_transform(self, x):
        """
        逆转换数据
        :param X: 转换数据
        :return: 转换后的数据
        """
        return np.dot(self.P.T, x.T).T


if __name__ == "__main__":
    normalData = pd.read_csv(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\正常数据.txt", sep=';').values[:, 1:].astype(np.float64)
    fault = pd.read_csv(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\一环路冷却剂流量-0.001-噪声波动.txt", sep=',').values[:, 1:].astype(np.float64)
    normal_testData = pd.read_excel(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\100%normal.xlsx").values[:, 1:].astype(np.float64)
    akr = PrincipalComponentAnalsys(n_component=3, contribution_rate=0.85, is_rate=False, is_normalization=True, reconstruction_error=1e-11)
    akr.fit(normalData)
    akr.save_model(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\结果\AAKR\AAKR.pkl')
    akr = PrincipalComponentAnalsys.load_model(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\结果\AAKR\AAKR.pkl')

    