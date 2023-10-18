from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Add
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR, LinearSVR
import joblib
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


class CNNReconstruction:
    def __init__(self, lr, batch_size, epochs, is_normalization=True, is_saveModel=False, normalization_range=(0, 1)):
        """
        Parameters
        ----------
        lr : float, 学习率，取值范围：0-1，一般为0.001
        batch_size ： int, 每批次训练数据量，一般为：50
        epochs : int, 训练次数，一般为：100
        """
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_normalization = is_normalization
        self.is_saveModel = is_saveModel
        self.normalization_range = normalization_range
        self.optimizer = Adam(learning_rate=self.lr)
        self.lossFunction = MeanSquaredError()

    def __normalization(self, X):
        """
        数据归一化
        :param X: 数据
        :return: 归一化后的数据
        """
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        return X, scaler

    def __builtModel(self):
        """构建模型"""
        model = Sequential()
        model.add(Conv2D(8, (1, 2), activation='selu', padding='same'))
        model.add(MaxPooling2D((1, 2), padding='same'))
        model.add(Conv2D(16, (1, 2), activation='selu', padding='same'))
        model.add(MaxPooling2D((1, 2), padding='same'))
        model.add(Conv2D(64, (1, 2), activation='selu', padding='same'))
        model.add(MaxPooling2D((1, 2), padding='same'))
        model.add(Conv2D(128, (1, 2), activation='selu', padding='same'))
        model.add(MaxPooling2D((1, 2), padding='same'))
        model.add(Conv2D(64, (1, 2), activation='selu', padding='same'))
        model.add(MaxPooling2D((1, 2), padding='same'))
        model.add(Conv2D(32, (1, 2), activation='selu', padding='same'))
        model.add(MaxPooling2D((1, 2), padding='same'))
        model.add(Conv2D(16, (1, 2), activation='selu', padding='same'))
        model.add(MaxPooling2D((1, 2), padding='same'))
        model.add(Conv2D(8, (1, 2), activation='selu', padding='same'))
        model.add(MaxPooling2D((1, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='selu'))
        model.add(Dense(64, activation='selu'))
        model.add(Dense(32, activation='selu'))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def fit(self, X, Y):
        """
        Parameters
        ----------
        :param X:
        :param Y:
        :return:
        """
        # 数据归一化
        if self.is_normalization:
            X, self.X_scaler = self.__normalization(X)
            Y, self.Y_scaler = self.__normalization(Y)
        # 转换数据格式
        X = X.reshape(X.shape[0], 1, X.shape[1], 1)
        self.model = self.__builtModel()
        self.model.compile(optimizer=self.optimizer, loss=self.lossFunction)
        self.model.fit(X, Y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        # 存储模型
        if self.is_saveModel:
            save_model(self.model, "CNNReconstruction.h5")
        tf.compat.v1.reset_default_graph()

    def predict(self, x, y=None):
        """
        Parameters
        ----------
        :param x:
        :return:
        """
        if self.is_normalization:
            x, self.x_scaler = self.__normalization(x)
            _, self.y_scaler = self.__normalization(np.array(y).reshape([-1, 1]))
        if self.is_saveModel:
            self.model = load_model("CNNReconstruction.h5")
        x = x.reshape(x.shape[0], 1, x.shape[1], 1)
        y_predict = self.model.predict(x)
        if self.is_normalization:
            y_predict = self.y_scaler.inverse_transform(np.array(y_predict).reshape([-1, 1]))
        tf.compat.v1.reset_default_graph()
        return y_predict