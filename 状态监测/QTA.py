import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.optimize import leastsq
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import re
from sklearn.preprocessing import MinMaxScaler

state = []
countList = []
suspiList = []
k1 = []


class QTA:
    def calc_sensi_threshold(self, normalData):
        sensiThreshold = {}
        n = np.shape(normalData)[0]
        normalDataMean = np.mean(normalData, axis=0)
        normalDataMeanRep = np.tile(normalDataMean, (n,1))
        normalDataStd = np.std(normalData, axis=0)
        normalDataMin = np.min(normalData, axis=0)
        normalDataMax = np.max(normalData, axis=0)
        if normalDataStd.any() == 0:
            normalDataStd += 1e-10
        g = np.sum((normalData-normalDataMeanRep)**3, axis=0)/((n-1)*normalDataStd**3)
        k = np.sum((normalData-normalDataMeanRep)**4, axis=0)/((n-1)*normalDataStd**4) - 3
        U1 = 1.96*(6*(n-2)/((n+1)*(n+3)))**0.5
        U2 = 1.96*(24*n*(n-2)*(n-3)/(((n+1)**2)*(n+3)*(n+5)))**0.5
        # calculate sensitive threshold
        for i in range(np.shape(normalData)[1]):
            sensiThreshold[i] = {}
            if (abs(g[i]) < U1) and (abs(k[i]) < U2):
                sensiThreshold[i]['low'] = normalDataMean[i] - 3*normalDataStd[i]
                sensiThreshold[i]['high'] = normalDataMean[i] + 3*normalDataStd[i]
            else:
                sensiThreshold[i]['low'] = normalDataMin[i]
                sensiThreshold[i]['high'] = normalDataMax[i]
        return sensiThreshold

    def func(self, p, x):
        k,b = p
        return k*x+b

    def error(self, p, x, y):
        return self.func(p,x)-y

    def QTA_monitor(self, step):
        global state, countList, suspiList, k1
        initValue = [1,2]
        para = [0]*len(countList)
        x = np.arange(len(suspiList))
        y = np.arange(len(suspiList), dtype=np.float)
        for i in range(len(countList)):
            if abs(countList[i]) >= step:
                for j in range(0, step):
                    y[j] = suspiList[j]['value'][i]
                para = leastsq(self.error, initValue, args=(x, y))
                k, b = para[0]
                if (k >= 0) and (countList[i] > 0):
                    state[i] = 1
                elif (k <= 0) and (countList[i] < 0):
                    state[i] = -1
                elif (k >= 0) and (countList[i] < 0):
                    state[i] = 0
                else:
                    state[i] = 0
                if countList[i] < 0:
                    countList[i] += 1
                else:
                    countList[i] -= 1
        del(suspiList[0:(len(suspiList)-step+1)])
        return state

    def QTA_threshold_monitor(self, X, index, sensiThreshold, step, determine_threshold):
        global state, countList, suspiList
        state = np.array(range(len(X)))
        for i in range(len(X)):
            if (X[i] > sensiThreshold[i]['high']) and (X[i] < determine_threshold[i]['high']):
                state[i] = 2
                countList[i] += 1
            elif (X[i] < sensiThreshold[i]['low']) and (X[i] > determine_threshold[i]['low']):
                state[i] = -2
                countList[i] -= 1
            elif X[i] > determine_threshold[i]['high']:
                state[i] = 1
            elif X[i] < determine_threshold[i]['low']:
                state[i] = -1
            else:
                state[i] = 0
                countList[i] = 0

        if (2 in state) or (-2 in state):
            suspiInfo = {}
            suspiInfo['index'] = index
            suspiInfo['state'] = state
            suspiInfo['value'] = X
            suspiList.append(suspiInfo)

        # 一场次数大于等于5，启动QTA
        for count in countList:
            if abs(count)>=5:
                state = self.QTA_monitor(step)
        return state

    def plot_QTA_fig(self, x, y, name, sensiThreshold, i, determine_threshold, plt_label):
        plt.clf()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        xminorLocator = MultipleLocator(5)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.xaxis.grid(True, which='major',linestyle='-.')
        ax.yaxis.grid(True, which='major',linestyle='-.')
        if plt_label == 0:
            plt.scatter(x, y, c='blue', marker='x', s=4)
        if plt_label == 1:
            plt.plot(x, y, c="b")
        if plt_label == 0:
            plt.yticks((-2, -1, 0, 1, 2))
        if plt_label == 1:
            plt.plot([sensiThreshold[i]["low"]]*len(x), c="y", label='敏感阈值')
            plt.plot([sensiThreshold[i]["high"]]*len(x), c="y")
            plt.plot([determine_threshold[i]["low"]]*len(x), c="r", label='确定阈值')
            plt.plot([determine_threshold[i]["high"]]*len(x), c="r")
        plt.xlabel(u'测试样本')
        plt.ylabel(u'监测结果')
        plt.title(u'QTA阈值法监测' + "(" + name + ")" + "特征")
        plt.legend()
        plt.show()

    def cal_determine_threshold(self, normalData):
        # 确定阈值
        scaler = MinMaxScaler()
        scaler.fit(normalData)
        determine_threshold = [0] * normalData.shape[1]
        for i in range(normalData.shape[1]):
            determine_threshold[i] = {}
            if scaler.data_min_[i] > 0:
                determine_threshold[i]["low"] = scaler.data_min_[i] * 0.98
            else:
                determine_threshold[i]["low"] = scaler.data_min_[i] * 1.02
            if scaler.data_max_[i] > 0:
                determine_threshold[i]["high"] = scaler.data_max_[i] * 1.02
            else:
                determine_threshold[i]["high"] = scaler.data_max_[i] * 0.98
        return determine_threshold

    def fit(self, normalData):
        """
        训练阶段，计算敏感阈值和确定阈值
        :param normalData:  正常数据
        :return:
        """
        normalData = np.array(normalData, dtype=np.float)
        self.sensiThreshold = self.calc_sensi_threshold(normalData)
        self.determineThreshold = self.cal_determine_threshold(normalData)

    def predict(self, testDataQTA, step):
        """

        :param testDataQTA:  测试数据
        :param step:  一般为5， 连续5个点超过阈值才会启动QTA
        :return:  返回测试数据的状态，±1表示特征数据在敏感阈值与确定阈值之间，0表示正常，±2表示特征数据超出确定阈值
        """
        global countList, state, suspiList
        states = []
        m_QTA = np.shape(testDataQTA)[0]
        n_QTA = np.shape(testDataQTA)[1]
        countList = [0]*n_QTA
        finalState = np.array([0]*m_QTA)
        for i in range(m_QTA):
            state = self.QTA_threshold_monitor(testDataQTA[i,:], i, self.sensiThreshold, step, self.determineThreshold)
            states.append(state)
        return np.array(states)



