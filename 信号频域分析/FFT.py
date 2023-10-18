import numpy as np


def FFT (Fs,data):
    L = len (data)                          # �źų���
    N = int(np.power(2,np.ceil(np.log2(L))))      # ��һ�����������
    FFT_y = (np.fft.fft(data,N))/L*2                  # N��FFT��������ʵ���źų��� L
    Fre = np.arange(int(N/2))*Fs/N          # Ƶ������
    FFT_y = FFT_y[range(int(N/2))]          # ȡһ��
    return Fre, np.abs(FFT_y)