import PyEMD
import pywt
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
import tqdm

# 显示中文以及负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 小波阈值函数去噪
class WaveletDenoising:
    def __init__(self, waveletName="db4", thresholdName="VisuShrink", thresholdFunctionName="soft", level=5,
                 mode="symmetric"):
        self.waveletName = waveletName
        self.thresholdName = thresholdName
        self.thresholdFunctionName = thresholdFunctionName
        self.level = level
        self.mode = mode

    """
    获取小波包阈值，实现一下阈值
    1、'rigsure' — Use the principle of Stein's Unbiased Risk.
    2、'heursure' — Use a heuristic variant of Stein's Unbiased Risk.
    3、'sqtwolog — Use the universal threshold √2ln(length(x)).
    4、'minimaxi' — Use minimax thresholding.
    5、'VisuShrink'— 通用阈值方法
    """

    def getThreshold(self, coeffs: np.ndarray) -> float:
        # 阈值选取
        threshold = 0
        if self.thresholdName == "minimaxi":
            if len(coeffs) >= 32:
                threshold = 0.3936 + 0.1829 * (np.log(len(coeffs)) / np.log(2))
            else:
                threshold = 0

        if self.thresholdName == "sqtwolog":
            threshold = np.sqrt(2 * np.log(len(coeffs)))

        if self.thresholdName == "rigsure":
            eta = (np.sum(np.square(coeffs)) - len(coeffs)) / len(coeffs)
            criti = np.sqrt((1 / len(coeffs) * (np.log(len(coeffs)) / np.log(2))))
            if eta < criti:
                threshold = np.sqrt(2 * np.log2(len(coeffs)))
            else:
                f = np.square(np.sort(np.abs(coeffs)))
                rish = []
                for i in range(len(f)):
                    rish.append(
                        (len(coeffs) - 2 * i + np.sum(f[:i]) + (len(coeffs) - i) * f[len(coeffs) - i - 1]) / len(
                            coeffs))
                indiex = np.argmin(rish)
                delta1 = np.sqrt(f[indiex])

                threshold = min([np.sqrt(2 * np.log2(len(coeffs))), delta1])

        if self.thresholdName == "heursure":
            f = np.square(np.sort(np.abs(coeffs)))
            rish = []
            for i in range(len(f)):
                rish.append(
                    (len(coeffs) - 2 * i + np.sum(f[:i]) + (len(coeffs) - i) * f[len(coeffs) - i - 1]) / len(coeffs))
            indiex = np.argmin(rish)
            threshold = np.sqrt(f[indiex])

        if self.thresholdName == "VisuShrink":
            threshold = np.median(np.abs(coeffs)) / 0.6745

        return threshold

    # 小波阈值函数
    def thresholdFunction(self, coeffs: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        thresholdMethod = {"soft": lambda: pywt.threshold(coeffs, threshold, "soft"),
                           "hard": lambda: pywt.threshold(coeffs, threshold, "hard"),
                           "greater": lambda: pywt.threshold(coeffs, threshold, "greater"),
                           "less": lambda: pywt.threshold(coeffs, threshold, "less"),
                           "": lambda: coeffs}
        assert self.thresholdFunctionName in thresholdMethod.keys(), f"No {self.thresholdFunctionName} threshold, current threshold have [soft, hard, greater, less]"
        coeffs = thresholdMethod[self.thresholdFunctionName]()
        return coeffs

    # 获取小波分解细节系数并输入到阈值函数中进行阈值去噪处理，处理后进行小波重构，最后得到去噪后的数据
    def waveletDecompose(self, signal: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec(signal, self.waveletName, level=self.level, mode=self.mode)
        for i in range(1, len(coeffs)):
            coeffs[i] = self.thresholdFunction(coeffs[i], self.getThreshold(coeffs[i]))
        denoiseSignal = pywt.waverec(coeffs, self.waveletName, mode=self.mode)
        return denoiseSignal


class EMDDenoising:
    def __init__(self,
                 denoiseMethodName="EED",
                 methodName="EMD",
                 thresholdName="VisuShrink",
                 T: Optional[np.ndarray] = None,
                 max_imf: int = -1,
                 progress: bool = False,
                 thresholdACF=0.5,
                 exceedThresholdNumber=5,
                 waveletName="db4",
                 thresholdFunctionName="soft",
                 level=5,
                 mode="symmetric"):
        """
        :param denoiseMethodName: 信号去噪方法: EED, ECD
        :param methodName: 信号分解方法: EMD, EEMD, CEEMDAN
        :param thresholdName: 阈值方法: VisuShrink, minimaxi, sqtwolog, rigsure, heursure
        :param T: EMD分解参数, 时间序列
        :param max_imf: EMD分解参数，最大分解层数
        :param progress: EMD分解参数，是否显示分解进度
        :param thresholdACF: EMD分解参数，阈值ACF
        :param exceedThresholdNumber: EMD分解参数, 超过阈值的次数
        :param waveletName: 小波分解参数, 小波基函数
        :param thresholdFunctionName: 阈值函数: soft, hard, greater, less
        :param level: 小波分解参数, 分解层数
        :param mode: 小波分解参数, 边界处理方式: symmetric, periodization, smooth, constant, reflect,antisymmetric, zero
        """
        self.denoiseMethodName = denoiseMethodName
        self.methodName = methodName
        self.thresholdName = thresholdName
        self.T = T
        self.max_imf = max_imf
        self.progress = progress
        self.thresholdACF = thresholdACF
        self.exceedThresholdNumber = exceedThresholdNumber
        self.waveletName = waveletName
        self.thresholdFunctionName = thresholdFunctionName
        self.level = level
        self.mode = mode

    # 信号分解
    def signalDecompose(self, signal: np.ndarray) -> np.ndarray:
        decomposeMethod = {"EMD": lambda: PyEMD.EMD().emd(S=signal, T=self.T, max_imf=self.max_imf),
                           "EEMD": lambda: PyEMD.EEMD().eemd(S=signal, T=self.T, max_imf=self.max_imf,
                                                             progress=self.progress),
                           "CEEMDAN": lambda: PyEMD.CEEMDAN().ceemdan(S=signal, T=self.T, max_imf=self.max_imf,
                                                                      progress=self.progress)}
        assert self.methodName in decomposeMethod.keys(), f"No {self.methodName} method, current method have [EMD, EEMD, CEEMDAN]"
        imfs = decomposeMethod[self.methodName]()
        return imfs

    # 计算energy后返回值的第一个局部最小值
    def calEnergyMin(self, imfs: np.ndarray) -> int:
        energy = np.mean(np.square(imfs), axis=1)
        # print(f"EnergyOfEachIMF={energy}")
        for index in range(1, len(energy) - 1):
            if energy[index] < energy[index - 1] and energy[index] < energy[index + 1]:
                # print(f"minValue of EnergyOfEachIMF is {energy[index]}")
                return index
        minIndex = np.argmin(energy)
        # print(f"minValue of EnergyOfEachIMF is {energy[minIndex]}")
        return minIndex

    # 计算calEnergyMin后返回值，并将其之后的IMF加和并返回
    def calEnergyIMFsSum(self, imfs: np.ndarray) -> np.ndarray:
        minIndex = self.calEnergyMin(imfs)
        return np.sum(imfs[minIndex + 1:, :], axis=0)

    # 基于能量信号的EMD去噪
    def EED(self, signal: np.ndarray) -> np.ndarray:
        imfs = self.signalDecompose(signal=signal)
        denoise = self.calEnergyIMFsSum(imfs)
        return denoise

    # 计算每个IMF的各阶数的自相关系数
    def calACF(self, imfs: np.ndarray) -> List[np.ndarray]:
        acf = []
        if self.progress:
            for i in tqdm.tqdm(range(imfs.shape[0]), desc="计算自相关系数"):
                acf.append(np.correlate(imfs[i, :], imfs[i, :], mode='same'))
        else:
            for i in range(imfs.shape[0]):
                acf.append(np.correlate(imfs[i, :], imfs[i, :], mode='same'))
        return acf

    # 根据ACF判断每个IMF是否是噪声占主导地位
    def judgeIMFIsNoise(self, acf: np.ndarray) -> bool:
        number = 0
        acf = np.sort(np.abs(acf))[::-1]
        for i in range(1, len(acf)):
            if number > self.exceedThresholdNumber:
                return False
            if acf[i] > self.thresholdACF * acf[0]:
                number += 1
        return True

    # 定义一个函数，利用judgeIMFIsNoise判断每个IMF是否是噪声信号占主导地位，若是则将信号进行小波去噪，去噪后将每个IMF相加获得去噪后的信号,基于自相关函数的去在方法
    def ECD(self, signal: np.ndarray) -> np.ndarray:
        imfs = self.signalDecompose(signal=signal)
        acf = self.calACF(imfs)
        for i in range(len(acf)):
            if self.judgeIMFIsNoise(acf[i]):
                imfs[i, :] = WaveletDenoising(waveletName=self.waveletName, thresholdName=self.thresholdName,
                                              thresholdFunctionName=self.thresholdFunctionName
                                              , level=self.level, mode=self.mode).waveletDecompose(imfs[i, :])
        denoise = np.sum(imfs, axis=0)
        return denoise

    def fit_transfrom(self, signal: np.ndarray) -> np.ndarray:
        denoiseMethod = {"EED": lambda: self.EED(signal=signal),
                         "ECD": lambda: self.ECD(signal=signal)}
        assert self.denoiseMethodName in denoiseMethod.keys(), f"No {self.denoiseMethodName} method, current method have [EED, ECD]"
        denoise = denoiseMethod[self.denoiseMethodName]()
        return denoise

    # 定义一个函数画出各个IMF自相关系数图
    def plotACF(self, acf, figsize=(10, 6)):
        plt.figure(figsize=figsize)
        for i in range(len(acf)):
            plt.subplot(len(acf), 1, i + 1)
            plt.plot(acf[i])
            plt.ylabel("ACF")
            plt.xlabel("Time [s]")
        plt.show()

    # 定义一个函数用于显示每个IMF的分解结果在一个图上，采用matplotlib方法绘制
    def plot_imfs(self, imfs, time, figsize=(10, 6)):
        plt.figure(figsize=figsize)
        for i in range(imfs.shape[0]):
            plt.subplot(imfs.shape[0], 1, i + 1)
            plt.plot(time, imfs[i, :], 'r')
            plt.ylabel("Amplitude")
            plt.xlabel("Time [s]")
        plt.show()


