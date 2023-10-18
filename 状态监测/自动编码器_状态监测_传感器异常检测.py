import os
import pandas as pd
from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from SensorFaultDetect import SensorFaultDetect
from 传感器定位柱状图 import sensor_location_bar_chart
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


class AutoEncoder(SensorFaultDetect):
    def __init__(self, lr=0.001, batch_size=50, is_normalize=True, is_add_noise=True, std=0.1):
        """
        Parameters
        ----------
        lr : float, 学习率，取值范围：0-1，一般为0.001
        batch_size ： int, 每批次训练数据量，一般为：50
        is_normalize : bool, 是否归一化，一般为：True
        is_add_noise : bool, 是否添加噪声，一般为：False
        std : float, 噪声标准差，一般为：0.1
        """
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = Adam(learning_rate=self.lr)
        self.is_normalize = is_normalize
        self.is_add_noise = is_add_noise
        self.std = std
        self.lossFunction = MeanSquaredError()
        
    def __normalize(self, X):
        """数据归一化

        Args:
            X (np.ndarray): data

        Returns:
            np.ndarray: normalized data
        """
        self.mms = MinMaxScaler()
        X = self.mms.fit_transform(X)
        return X

    def __encoder(self, shape):
        # 时域网络-编码器
        model = Sequential()
        # model.add(Dense(shape[0], activation="selu"))
        # model.add(Dense(shape[1], activation="selu"))
        # model.add(Dense(shape[2], activation="selu"))
        # 采用卷积神经网络进行编码
        model.add(Dense(50, activation="selu"))
        model.add(Reshape((-1, 1, 50)))
        model.add(Conv2D(filters=32, kernel_size=(1, 3), activation="selu", padding="same"))
        model.add(MaxPooling2D(pool_size=(1, 2), padding="same"))
        model.add(Conv2D(filters=16, kernel_size=(1, 3), activation="selu", padding="same"))
        model.add(MaxPooling2D(pool_size=(1, 2), padding="same"))
        model.add(Conv2D(filters=8, kernel_size=(1, 3), activation="selu", padding="same"))
        model.add(Flatten())
        model.add(Dense(30, activation="selu", input_shape=(shape[1], 1)))
        model.add(Dense(15, activation="selu", input_shape=(shape[1], 1)))
        plot_model(model, "encoder.png", show_shapes=True, show_dtype=True)
        return model

    def __decoder(self, X, shape):
        model = Sequential()
        # model.add(Dense(shape[1], activation="selu"))
        # model.add(Dense(shape[0], activation="selu"))
        # model.add(Dense(X.shape[1], activation="sigmoid"))
        # 采用卷积神经网络进行解码
        model.add(Dense(30, activation="selu"))
        model.add(Reshape((-1, 1, 30)))
        model.add(Conv2DTranspose(filters=8, kernel_size=(1, 3), activation="selu", padding="same"))
        model.add(MaxPooling2D(pool_size=(1, 2), padding="same"))
        model.add(Conv2DTranspose(filters=16, kernel_size=(1, 3), activation="selu", padding="same"))
        model.add(MaxPooling2D(pool_size=(1, 2), padding="same"))
        model.add(Conv2DTranspose(filters=32, kernel_size=(1, 3), activation="selu", padding="same"))
        model.add(Flatten())
        model.add(Dense(50, activation="selu"))
        model.add(Dense(X.shape[1], activation="sigmoid"))
        return model

    def __modelbuilt(self, X, shape):
        """构建模型"""
        input = Input(shape=(X.shape[1], ))
        self.encoderModel = self.__encoder(shape)
        middle = self.encoderModel(input)
        self.decoderModel = self.__decoder(X, shape)
        output = self.decoderModel(middle)
        model = Model(inputs=input, outputs=output)

        return model

    def fit(self, X, shape, epoch, model_path):
        """
        Parameters
        ----------
        X : array, 训练数据
        shape ： [list, tuple, array], 中间层神经元个数，一般为：[50， 1]
        epoch : int, 训练次数，一般为：10000
        seed ： int, 随机数种子，一般为：52
        model_path ： string, 模型保存地址

        Returns
        -------
        loss1 ： float, 训练误差
        DownLimit ：float, 阈值下限
        UpLimit ：float, 阈值上限
        """
        # 数据归一化
        if self.is_normalize:
            X = self.__normalize(X)
        # 添加噪声
        if self.is_add_noise:
            X = X + np.random.normal(0, self.std, X.shape)
        self.__x = X
        # 构建模型
        model = self.__modelbuilt(X, shape)
        model.compile(optimizer=self.optimizer, loss=self.lossFunction)
        model.fit(X, X, batch_size=self.batch_size, epochs=epoch, verbose=1)
        self.loss = model.history.history["loss"]
        # 存储模型
        save_model(model, f"{model_path}model_AutoEncoder.h5")
        save_model(self.encoderModel, f"{model_path}encoderModel.h5")
        save_model(self.decoderModel, f"{model_path}decoderModel.h5")
        joblib.dump(self.mms, f"{model_path}mms.pkl")
        tf.compat.v1.reset_default_graph()
    
    def threshold(self, model_path, x=None):
        # 计算自编码器输出与输入欧氏距离
        if self.is_normalize:
            mms = joblib.load(f"{model_path}mms.pkl")
        # 生成网络
        loadModel = load_model(f"{model_path}model_AutoEncoder.h5", compile=False)
        if x is None:
            result = loadModel.predict(self.__x)
            distance = self.__calcuateDistance(result, self.__x)
        else:
            # 数据归一化
            if self.is_normalize:
                x = mms.transform(x)
            result = loadModel.predict(x)
            distance = self.__calcuateDistance(result, x)
        # 计算阈值
        up_threshold, down_threshold = self._box_plot(distance.reshape(-1, 1), is_zero=False)
        return up_threshold, down_threshold

    def __calcuateDistance(self, x, y):
        temp = (x - y)**2
        distance = np.sqrt(np.sum(temp, axis=1))
        return distance

    def predict(self, X, model_path):
        """
        Parameters
        ----------
        X : array, 训练数据
        model_path : string, 模型保存地址

        Returns
        -------
        re : float, 测试结果
        """
        # 数据归一化
        if self.is_normalize:
            mms = joblib.load(f"{model_path}mms.pkl")
            X = mms.transform(X)
        # 生成网络
        loadModel = load_model(f"{model_path}model_AutoEncoder.h5", compile=False)
        result = loadModel.predict(X)
        distance = self.__calcuateDistance(result, X)
        tf.compat.v1.reset_default_graph()
        return distance

    def sensor_fault_location_threshold(self, model_path, X=None):
        if X is None:
            X = self.__x
        feature_rate = self.__predict_feature_rate(X, model_path)
        # 计算阈值
        up_threshold, down_threshold = self._box_plot(feature_rate, is_zero=True)
        return up_threshold, down_threshold

    def sensor_fault_location_predict(self, x, model_path):
        # 数据归一化
        if self.is_normalize:
            mms = joblib.load(f"{model_path}mms.pkl")
            x = mms.transform(x)
        # 生成网络
        feature_rate = self.__predict_feature_rate(x, model_path)
        return feature_rate

    def __predict_feature_rate(self, x, model_path):
        loadModel = load_model(f"{model_path}model_AutoEncoder.h5", compile=False)
        result = loadModel.predict(x)
        residual = np.abs(x - result)
        feature_rate = residual / np.sum(residual, axis=1, keepdims=True)
        return feature_rate
    
    def save_model(self, file_save_path):
        pass
    
    @staticmethod
    def load_model(self, file_save_path):
        pass 


if __name__ == "__main__":
    normalData = pd.read_csv(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\正常数据.txt", sep=';').values[:, 1:].astype(np.float64)
    fault = pd.read_csv(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\一环路冷却剂流量-0.001-噪声波动.txt", sep=',').values[:, 1:].astype(np.float64)
    normal_testData = pd.read_excel(r"D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\数据\100%normal.xlsx").values[:, 1:].astype(np.float64)
    akr = AutoEncoder(lr=0.001, batch_size=50, is_normalize=True, is_add_noise=True, std=0.05)
    akr.fit(normalData, [50, 15], 100, r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\\结果\AE\\')
    # akr.save_model(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\结果\AE\AE.pkl')
    # akr = AutoEncoder.load_model(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\代码\结果\AE\AE.pkl')
    # 异常检测
    result = akr.predict(fault, r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\\代码\\结果\\AE\\')
    up_threshold, down_threshold = akr.threshold(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\\代码\\结果\\AE\\')
    plt.plot(result)
    plt.axhline(up_threshold, color='r', linestyle='--')
    plt.axhline(down_threshold, color='r', linestyle='--')
    plt.title('异常检测')
    plt.xlabel('数据编号')
    plt.ylabel('异常指标')
    plt.show()
    
    # 计算阈值
    up_threshold, down_threshold = akr.sensor_fault_location_threshold(r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\\代码\\结果\\AE\\')
    result = akr.sensor_fault_location_predict(fault, r'D:\文件\小论文\基于D-S证据理论的传感器故障定位方法研究\\代码\\结果\\AE\\')
    # 画出每一行柱状图
    # 计算每一个传感器的异常指标是否超过阈值，若超出阈值则认为该传感器异常, color为异常传感器的颜色red, 否则为green
    for i in range(result.shape[0]):
        # 画出每一行柱状图
        sensor_location_bar_chart(result[i, :].reshape(1, -1), up_threshold, down_threshold, title=f'MEST-故障传感器-{i}', xlabel='传感器编号', ylabel='传感器数量')
        plt.savefig(f'D:/文件/小论文/基于D-S证据理论的传感器故障定位方法研究/代码/结果/AE/AE-故障传感器-{i}.png')
        plt.close()
