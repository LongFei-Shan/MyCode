import pywt
from typing import Optional, List
import numpy as np


class WaveletDenoising:
    def __init__(self, waveletName="db4", thresholdName="VisuShrink", thresholdFunctionName="soft", level=5,
                 mode="symmetric"):
        """

        :param waveletName:  小波基函数
        :param thresholdName:  阈值方法: VisuShrink, minimaxi, sqtwolog, rigsure, heursure
        :param thresholdFunctionName:  阈值函数: soft, hard, greater, less
        :param level:  分解层数
        :param mode:  边界处理方式: symmetric, periodization, smooth, constant, reflect,antisymmetric, zero
        """
        self.waveletName = waveletName
        self.thresholdName = thresholdName
        self.thresholdFunctionName = thresholdFunctionName
        self.level = level
        self.mode = mode

    def __getThreshold(self, coeffs: np.ndarray) -> float:
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
    def __thresholdFunction(self, coeffs: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        thresholdMethod = {"soft": lambda: pywt.threshold(coeffs, threshold, "soft"),
                           "hard": lambda: pywt.threshold(coeffs, threshold, "hard"),
                           "greater": lambda: pywt.threshold(coeffs, threshold, "greater"),
                           "less": lambda: pywt.threshold(coeffs, threshold, "less"),
                           "": lambda: coeffs}
        assert self.thresholdFunctionName in thresholdMethod.keys(), f"No {self.thresholdFunctionName} threshold, current threshold have [soft, hard, greater, less]"
        coeffs = thresholdMethod[self.thresholdFunctionName]()
        return coeffs

    # 获取小波分解细节系数并输入到阈值函数中进行阈值去噪处理，处理后进行小波重构，最后得到去噪后的数据
    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """

        :param signal:  输入信号
        :return:
        """
        coeffs = pywt.wavedec(signal, self.waveletName, level=self.level, mode=self.mode)
        for i in range(1, len(coeffs)):
            coeffs[i] = self.__thresholdFunction(coeffs[i], self.__getThreshold(coeffs[i]))
        denoiseSignal = pywt.waverec(coeffs, self.waveletName, mode=self.mode)
        return denoiseSignal




