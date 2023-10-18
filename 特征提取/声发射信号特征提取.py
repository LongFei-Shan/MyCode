import numpy as np
from scipy.signal import find_peaks
from scipy.signal import welch


class AcousticEmissionFeatureExtractor:
    def __init__(self, sampling_rate):
        """
        Parameters
        :param sampling_rate:  采样率
        """
        self.sampling_rate = sampling_rate

    def amplitude(self, signal):
        """
        计算信号的振幅
        :param signal:  信号
        :return:
        """
        return np.max(np.abs(signal))

    def energy(self, signal):
        """
        计算信号的能量
        :param signal:  信号
        :return:
        """
        return np.sum(signal ** 2)

    def rms(self, signal):
        """
        计算信号的均方根
        :param signal:  信号
        :return:
        """
        return np.sqrt(np.mean(signal ** 2))

    def ring_count(self, signal, threshold):
        """
        振铃计数
        :param signal:  信号
        :param threshold:  阈值
        :return:
        """
        peaks, _ = find_peaks(signal, height=threshold)
        return len(peaks)

    def duration(self, signal):
        """
        持续时间
        :param signal:  信号
        :return:
        """
        return len(signal) / self.sampling_rate

    def rise_time(self, signal, threshold):
        peaks, _ = find_peaks(signal, height=threshold)
        if len(peaks) > 0:
            return (peaks[0] - np.where(signal < threshold)[0][-1]) / self.sampling_rate
        else:
            return np.nan

    def peak_count(self, signal, threshold):
        """
        峰值计数
        :param signal:   信号
        :param threshold:  阈值
        :return:
        """
        peaks, _ = find_peaks(signal, height=threshold)
        return len(peaks)

    def mean_frequency(self, signal):
        """
        均值频率
        :param signal:  信号
        :return:
        """
        freqs, psd = welch(signal, fs=self.sampling_rate)
        return np.average(freqs, weights=psd)

    def inverse_frequency(self, signal):
        """
        逆频率
        :param signal:  信号
        :param sampling_rate:  采样率
        :return:
        """
        return 1 / (np.argmax(np.abs(np.fft.fft(signal))) / (len(signal) / self.sampling_rate))

    def initial_frequency(self, signal):
        """
        初始频率
        :param signal:  信号
        :return:
        """
        return np.argmax(np.abs(np.fft.fft(signal))) / (len(signal) / self.sampling_rate)

    def centroid_frequency(self, signal):
        """
        质心频率
        :param signal:  信号
        :return:
        """
        freqs, psd = welch(signal, fs=self.sampling_rate)
        return np.sum(freqs * psd) / np.sum(psd)

    def peak_frequency(self, signal, threshold):
        """
        峰值频率
        :param signal:  信号
        :param threshold:  阈值
        :param sampling_rate:
        :return:
        """
        peaks, _ = find_peaks(signal, height=threshold)
        return np.argmax(np.abs(np.fft.fft(signal[peaks]))) / (len(signal) / self.sampling_rate)

    def signal_strength(self, signal):
        """
        信号强度
        :param signal:  信号
        :return:
        """
        return np.sum(signal ** 2)

    def absolute_frequency(self, signal):
        """
        绝对频率
        :param signal:  信号
        :return:
        """
        return np.argmax(np.abs(np.fft.fft(signal))) / (len(signal) / self.sampling_rate)

    def absolute_energy(self, signal):
        """
        绝对能量
        :param signal:  信号
        :return:
        """
        return np.sum(np.abs(signal))

    def arrival_time(self, signal, threshold):
        """
        到达时间
        :param signal:
        :param threshold:
        :return:
        """
        peaks, _ = find_peaks(signal, height=threshold)
        if len(peaks) > 0:
            return peaks[0] / self.sampling_rate
        else:
            return np.nan

    def fit_transform(self, signal, threshold):
        """
        计算所有特征
        :param signal:  信号
        :param threshold:  阈值
        :return:
        """
        feature = np.array([self.amplitude(signal),
            self.energy(signal),
            self.rms(signal),
            self.ring_count(signal, threshold),
            self.duration(signal),
            self.rise_time(signal, threshold),
            self.peak_count(signal, threshold),
            self.mean_frequency(signal),
            self.inverse_frequency(signal),
            self.initial_frequency(signal),
            self.centroid_frequency(signal),
            self.peak_frequency(signal, threshold),
            self.signal_strength(signal),
            self.absolute_frequency(signal),
            self.absolute_energy(signal),
            self.arrival_time(signal, threshold)])
        return feature


