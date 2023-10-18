#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :SVD去噪.py
# @Time      :2023/9/5 15:44
# @Author    :LongFei Shan
from numpy.linalg import svd
import numpy as np
import matplotlib.pyplot as plt


def FFT (Fs,data):
    L = len (data)                          # 信号长度
    N = int(np.power(2,np.ceil(np.log2(L))))      # 下一个最近二次幂
    FFT_y = (np.fft.fft(data,N))/L*2                  # N点FFT，但除以实际信号长度 L
    Fre = np.arange(int(N/2))*Fs/N          # 频率坐标
    FFT_y = FFT_y[range(int(N/2))]          # 取一半
    return Fre, np.abs(FFT_y)


class SVD_Denoise:
    def __init__(self, size, component=3, full_matrices=True, compute_uv=True, hermitian=False):
        """

        :param size:  原始信号狗仔成矩阵的行与列数
        :param component:  保留的奇异值个数
        :param full_matrices: bool, optional
        If True (default), `u` and `vh` have the shapes ``(..., M, M)`` and
        ``(..., N, N)``, respectively.  Otherwise, the shapes are
        ``(..., M, K)`` and ``(..., K, N)``, respectively, where
        ``K = min(M, N)``.
        :param compute_uv:  bool, optional
        Whether or not to compute `u` and `vh` in addition to `s`.  True
        by default.
        :param hermitian: bool, optional
        If True, `a` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.
        """
        self.size = size
        self.component = component
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv
        self.hermitian = hermitian

    def calcuate_component(self, s):
        component = 0
        rate = 0
        for i in range(len(s)):
            if rate/sum(s) >= self.component:
                break
            rate += s[i]
            component += 1
        return component

    def fit_transform(self, x):
        assert np.array(x).ndim == 1, "输入信号维度不为1"
        assert len(x) == self.size[0]*self.size[1], "输入信号长度与构造矩阵长度不一致"
        # 构造矩阵
        x_reshape = np.array(x).reshape(self.size[0], self.size[1])
        # 去噪
        U, s, V = svd(x_reshape, self.full_matrices, self.compute_uv, self.hermitian)
        if len(s) < (self.size[0]):
            s = np.append(s, np.zeros((self.size[0])-len(s)))
        # 保留前self.component个奇异值
        if self.component < 1:
            self.component = self.calcuate_component(s)
        s[self.component:] = 0
        s = np.diag(s)[:, :self.size[1]]
        # 重构信号
        x_restore = U@s@V
        x_restore = x_restore.reshape(len(x))
        return x_restore

def svdDenoise():
    # 定义一个正弦函数
    length = 50000
    time = 1
    x = np.linspace(0, time, length)
    y = np.sin(200*x) + np.sin(500*x) + np.sin(1000*x) + np.sin(5000*x) + np.sin(6000*x)
    # 加入噪声
    y_noise = y + np.random.normal(0, 20, len(y))
    # SVD去噪
    svd = SVD_Denoise(size=(500, 100), component=0.9)
    y_restore = svd.fit_transform(y)
    # 绘图

    plt.plot(x, y, label="original")
    plt.plot(x, y_noise, label="with noise", alpha=0.5)
    plt.plot(x, y_restore, label="restored", alpha=0.5)
    plt.legend()
    plt.show()

    Fs = int(length/time)
    # 绘制频谱
    Fre, FFT_y = FFT(Fs, y)
    Fre, FFT_y_noise = FFT(Fs, y_noise)
    Fre, FFT_y_restore = FFT(Fs, y_restore)
    plt.plot(Fre, FFT_y, label="original")
    plt.plot(Fre, FFT_y_noise, label="with noise", alpha=0.5)
    plt.plot(Fre, FFT_y_restore, label="restored", alpha=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    svdDenoise()
