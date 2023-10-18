import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from 自动编码器_状态监测_传感器异常检测 import ds_AE
from PCA_状态监测_传感器异常检测 import ds_pca


class DSTheoryEvidence:
    def __init__(self, n_feature):
        self.n_feature = n_feature

    def calcuate_K(self, fault_prob):
        """计算K值"""
        K = np.sum(fault_prob[0]*fault_prob[1]*fault_prob[2])
        return K

    def calcuate_Mass(self, fault_prob, K):
        mass = fault_prob[0]*fault_prob[1]*fault_prob[2]
        mass = mass/K
        return mass

    def fit_transform(self, fault_prob):
        K = self.calcuate_K(fault_prob)
        mass = self.calcuate_Mass(fault_prob, K)
        return mass


if __name__ == "__main__":
    io = r"代码\正常数据\MoreDataTemp.txt"
    ds_AE(io, ';')
    ds_pca(io, ";")
    ds = DSTheoryEvidence(70)
    max_index = []
    max_index_pro = []
    for index in tqdm.tqdm(range(300), ncols=100, desc='计算中'):
        # 加载故障概率
        fault_prob_ae = pd.read_excel(r'代码\传感器故障概率\AE_feature_rate.xlsx', header=None).values[index, 1:]
        fault_prob_spe = pd.read_excel(r'代码\传感器故障概率\SPE_FeatureRate.xlsx', header=None).values[index, 1:]
        fault_prob_t2 = pd.read_excel(r'代码\传感器故障概率\T2_FeatureRate.xlsx', header=None).values[index, 1:]
        fault_prob_t2 = fault_prob_t2/np.sum(fault_prob_t2)
        fault_prob = np.array([fault_prob_ae, fault_prob_spe, fault_prob_t2])
        # fault_prob = np.delete(fault_prob, [1], axis=1)
        mass = ds.fit_transform(fault_prob)
        max_index.append(np.max(mass))
        max_index_pro.append(np.argmax(mass))

    plt.subplot(2, 1, 1)
    plt.title('DS-故障传感器')
    plt.plot(max_index)
    plt.subplot(2, 1, 2)
    plt.title('DS-故障传感器概率')
    plt.plot(max_index_pro)
    plt.tight_layout()
    plt.show()