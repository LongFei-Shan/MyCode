import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pywt
from sampen import sampen2


class VibrationSignalFeatureExtraction:
    timeFeatureName = ["平均值", "方差", "能量", "均方根", "标准差", "最大值", "最小值", "峰值系数", "峭度因子", "偏度因子", "波形系数", "脉冲因子", "裕度因子"]
    freqFeatureName = ["频率幅值平均值", "重心频率", "均方根频率", "标准差频率", "频率集中度", "频率峭度"]

    # 静态函数
    @staticmethod
    def FFT(Fs, data):
        L = len(data)  # 信号长度
        N = int(np.power(2, np.ceil(np.log2(L))))  # 下一个最近二次幂
        FFT_y = (np.fft.fft(data, N)) / L * 2  # N点FFT，但除以实际信号长度 L
        Fre = (np.arange(int(N / 2)) / N) * Fs  # 频率坐标
        FFT_y = FFT_y[:int(N / 2)]  # 取一半
        return Fre, np.abs(FFT_y)

    @staticmethod
    def freFeature(data, Fs):
        """频域特征"""
        try:
            data = np.array(data).ravel()
            Fre, FFT_y = VibrationSignalFeatureExtraction.FFT(Fs, data)
            # 频率幅值平均值
            S1 = np.sum(FFT_y) / len(FFT_y)
            # 重心频率
            S2 = np.sum(np.array(Fre) * np.array(FFT_y)) / np.sum(FFT_y)
            # 均方根频率
            S3 = np.sqrt(np.sum(np.array(FFT_y) * np.array(FFT_y)) / len(FFT_y))
            # 标准差频率
            S4 = np.sqrt(np.sum((np.array(FFT_y) - S1) * (np.array(FFT_y) - S1)) / len(FFT_y))
            # 频率集中度（向重心频率靠拢的集中度）
            S5 = 1 - (np.sum(np.abs(np.array(FFT_y) * np.array(Fre - S2))) / np.sum(np.array(FFT_y) * np.array(Fre)))
            # 频率峭度
            S6 = np.sum((np.array(FFT_y) - S1) ** 4) / (len(FFT_y) * (S4 ** 4))
            return np.array([S1, S2, S3, S4, S5, S6])
        except:
            return
    # [频率幅值平均值,重心频率,均方根频率,标准差频率,频率集中度,频率峭度]

    @staticmethod
    def timeFeature(subData):
        """信号特征"""
        try:
            subData = np.array(subData).ravel()
            # -------时域特征-------
            # 平均值
            Mean = np.mean(subData)
            # 方差
            Var = np.var(subData, ddof=1)
            # 平均幅值
            MeanAmplitude = np.mean(np.abs(subData))
            # 能量
            Energy = np.sum(np.power(subData, 2))
            # 均方根
            RMS = np.sqrt(np.mean(np.power(subData, 2)))
            # 方根幅值
            SquareRootAmplitude = np.power(np.mean(np.sqrt(np.abs(subData))), 2)
            # 标准差
            STD = np.std(subData, ddof=1)
            # 最大值
            Max = np.mean(np.sort(subData)[-10:])
            # 最小值
            Min = np.mean(np.sort(subData)[:10])

            # -------波形特征-------
            # 峰值
            Peak = np.mean(np.sort(subData)[-10:])
            # 峰值系数
            if RMS == 0:
                Cf = 0
            else:
                Cf = np.mean(np.abs(np.sort(subData)[-10:])) / RMS
            # 峭度
            Kurtosis = np.sum((subData - Mean) ** 4)
            # 峭度因子
            if STD == 0:
                KurtosisFactor = 0
                SkewnessFactor = 0
            else:
                KurtosisFactor = Kurtosis / ((STD ** 4) * (len(subData - 1)))
                # 偏度因子
                SkewnessFactor = np.sum((np.abs(subData) - Mean) ** 3) / ((STD ** 3) * (len(subData - 1)))
            # 波形系数
            if Mean == 0:
                Cs = 0
            else:
                Cs = RMS / Mean
            # 脉冲因子
            if MeanAmplitude == 0:
                ImpulseFactor = 0
            else:
                ImpulseFactor = Cf / MeanAmplitude
            # 裕度因子
            if SquareRootAmplitude == 0:
                MarginFactor = 0
            else:
                MarginFactor = Cf / SquareRootAmplitude

            return [Mean, Var, Energy, RMS, STD, Max, Min, Cf, KurtosisFactor, SkewnessFactor, Cs, ImpulseFactor,
                    MarginFactor]
        except:
            return
    # ["平均值", "方差", "能量", "均方根", "标准差", "最大值", "最小值", "峰值系数", "峭度因子", "偏度因子", "波形系数", "脉冲因子", "裕度因子"]

    @staticmethod
    def waveEnergy(data, wavelet, maxlevel):
        # 小波包能量
        try:
            data = np.array(data).ravel()
            wpt = pywt.WaveletPacket(data, wavelet=wavelet, maxlevel=maxlevel)
            subenergy = []
            for i in range(maxlevel, maxlevel + 1):
                node = wpt.get_level(i, "freq")
                for loop, sub_data in enumerate(node):
                    temp = np.sum(np.array(wpt[sub_data.path].data) ** 2)
                    subenergy.append(temp)
            return np.array(subenergy)
        except:
            return

    @staticmethod
    def SampenFeature(data, mm=2, r=0.2):
        # 样本熵
        try:
            data = np.array(data).ravel()
            result = sampen2(data, mm=mm, r=r)
            return np.array(result)
        except:
            return