import matplotlib.pyplot as plt
from PyEMD import CEEMDAN, EMD, EEMD
import numpy as np
from scipy.signal import hilbert
from vmdpy import VMD
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False


class HilbertHuangTransform:
    def __init__(self):
        pass

    def __SignalTransfrom(self, x, t=None, name="VMD", alpha=1000, tau=0, K=6, DC=0, init=1, tol=1e-7, max_imf=-1):
        imfs = ""
        if name == "EMD":
            emd = EMD()
            imfs = emd.emd(x, t, max_imf=max_imf)
        elif name == "EEMD":
            emd = EEMD()
            imfs = emd.eemd(x, t, max_imf=max_imf)
        elif name == "CEEMDAN":
            emd = CEEMDAN()
            imfs = emd.ceemdan(x, t, max_imf=max_imf)
        elif name == "VMD":
            u, u_hat, omega = VMD(f=x, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol)
            imfs = u
        else:
            ValueError(f"{name}不存在")

        return imfs

    def __HilbertTransform(self, imfs):
        """
        希尔伯特变换

        :param imfs: 信号的IMF
        :return:
        """
        analyticSignal = []
        for imf in imfs:
            temp = hilbert(imf)
            analyticSignal.append(temp)
        return np.array(analyticSignal)

    def __CalculateFrequency(self, fs, analyticSignal):
        frequency = []
        amplitude = []
        for signal in analyticSignal:
            # 计算瞬时相位
            angle = np.angle(signal)
            inst_phase = np.unwrap(angle)
            # 计算瞬时幅值
            inst_amp = np.abs(signal)
            amplitude.append(inst_amp)
            # 计算瞬时频率
            inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs
            frequency.append(inst_freq)
        return np.array(frequency), np.array(amplitude)

    def __HilbertSpectrum(self, fs, frequency, amplitude, f_bins=500):
        freq_edges = np.linspace(0, fs / 2, f_bins + 1)
        hs = np.zeros((f_bins, frequency.shape[1]))
        for i in range(len(frequency)):
            for j in range(len(frequency[i])):
                f_index = np.digitize(frequency[i][j], freq_edges) - 1
                if f_index >= 0 and f_index < f_bins:
                    hs[f_index, j] += amplitude[i][j]
        return hs, freq_edges

    def fit_transform(self, x, fs, t=None, name="VMD", alpha=1000, tau=0, K=5, DC=0, init=1, tol=1e-7, max_imf=5):
        """
        运行程序入口

        :param x: 输入信号
        :param fs: 采样频率
        :param t: 输入信号对应的时间
        :param name: 选择的模型，现有模型有：VMD,EMD,EEMD,CEEMDAN, 注：数据量大时，VMD，EEMD,CEEMDAN运行时间很长
        :param alpha: 带宽限制 经验取值为 抽样点长度 1.5-2.0 倍；仅对VMD有效
        :param tau: 噪声容限, 表示所有IMF加起来是否等于原始信号，如果等于0，表示必须等于原始信号，否则可以有误差，仅对VMD有效
        :param K: 分解模态（IMF）个数，仅对VMD有效
        :param DC: 合成信号若无常量，取值为 0；若含常量，则其取值为 1；仅对VMD有效
        :param init: 初始化 w 值，当初始化为 1 时，均匀分布产生的随机数；仅对VMD有效
        :param tol: 控制误差大小常量，决定精度与迭代次数，仅对VMD有效
        :param max_imf: EMD,EEMD,CEEMDAN最大分解IMF个数
        :return: frequency, imfs, hs, freq_edges, t
                frequency表示每个IMF的频率
                imfs表示分解的IMF
                hs表示希尔伯特谱值
                freq_edges表示希尔伯特谱值纵坐标
                t表示希尔伯特谱值横坐标
        """
        self.t = t
        imfs = self.__SignalTransfrom(x, t, name, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol, max_imf=max_imf)
        analyticSignal = self.__HilbertTransform(imfs)
        frequency, amplitude = self.__CalculateFrequency(fs, analyticSignal)
        self.hs, self.freq_edges = self.__HilbertSpectrum(fs, frequency, amplitude)
        return frequency, imfs, self.hs, self.freq_edges, t

    def plot_HilbertSpectrum(self):
        # 画出希尔伯特谱
        assert not (self.t is None), "t不能是None"
        # 归一化
        plt.figure()
        plt.pcolormesh(self.t, self.freq_edges, self.hs, cmap='jet', shading='auto', vmin=0, vmax=1, rasterized=True, antialiased=True)
        plt.title('Hilbert-Huang Transform')
        plt.ylabel('频率[Hz]')
        plt.xlabel('时间[s]')
        plt.tight_layout()
        plt.show()
