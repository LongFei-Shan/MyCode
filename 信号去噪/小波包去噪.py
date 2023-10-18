import pywt
from typing import Optional, List
import numpy as np


class WaveletDenoising:
    def __init__(self, waveletName="db4", thresholdName="VisuShrink", thresholdFunctionName="soft", level=5,
                 mode="symmetric"):
        """

        :param waveletName:  С��������
        :param thresholdName:  ��ֵ����: VisuShrink, minimaxi, sqtwolog, rigsure, heursure
        :param thresholdFunctionName:  ��ֵ����: soft, hard, greater, less
        :param level:  �ֽ����
        :param mode:  �߽紦��ʽ: symmetric, periodization, smooth, constant, reflect,antisymmetric, zero
        """
        self.waveletName = waveletName
        self.thresholdName = thresholdName
        self.thresholdFunctionName = thresholdFunctionName
        self.level = level
        self.mode = mode

    def __getThreshold(self, coeffs: np.ndarray) -> float:
        # ��ֵѡȡ
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

    # С����ֵ����
    def __thresholdFunction(self, coeffs: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        thresholdMethod = {"soft": lambda: pywt.threshold(coeffs, threshold, "soft"),
                           "hard": lambda: pywt.threshold(coeffs, threshold, "hard"),
                           "greater": lambda: pywt.threshold(coeffs, threshold, "greater"),
                           "less": lambda: pywt.threshold(coeffs, threshold, "less"),
                           "": lambda: coeffs}
        assert self.thresholdFunctionName in thresholdMethod.keys(), f"No {self.thresholdFunctionName} threshold, current threshold have [soft, hard, greater, less]"
        coeffs = thresholdMethod[self.thresholdFunctionName]()
        return coeffs

    # ��ȡС���ֽ�ϸ��ϵ�������뵽��ֵ�����н�����ֵȥ�봦����������С���ع������õ�ȥ��������
    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """

        :param signal:  �����ź�
        :return:
        """
        coeffs = pywt.wavedec(signal, self.waveletName, level=self.level, mode=self.mode)
        for i in range(1, len(coeffs)):
            coeffs[i] = self.__thresholdFunction(coeffs[i], self.__getThreshold(coeffs[i]))
        denoiseSignal = pywt.waverec(coeffs, self.waveletName, mode=self.mode)
        return denoiseSignal




