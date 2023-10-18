import numpy as np


def FFT (Fs,data):
    L = len (data)                          # 信号长度
    N = int(np.power(2,np.ceil(np.log2(L))))      # 下一个最近二次幂
    FFT_y = (np.fft.fft(data,N))/L*2                  # N点FFT，但除以实际信号长度 L
    Fre = np.arange(int(N/2))*Fs/N          # 频率坐标
    FFT_y = FFT_y[range(int(N/2))]          # 取一半
    return Fre, np.abs(FFT_y)