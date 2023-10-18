#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :LSTM故障诊断.py
# @Time      :2023/8/8 17:55
# @Author    :LongFei Shan

# LSTM分类模型，使用tensorflow2.0版本:
# 1.导入相关库
# 2、要有数据是否归一化的判断
# 3、__init__函数用于初始化模型参数，例如输入数据的shape，输出数据的shape，隐藏层的神经元个数，学习率，epoch, batch_size等
# 4、__build_model函数用于构建模型，例如：输入层，隐藏层，输出层，损失函数，优化器等
# 5、fit函数用于训练模型，例如：训练数据，训练标签，验证数据，验证标签等
# 6、predict函数用于预测，例如：测试数据，测试标签等
# 7、模型要有保存和加载模型的功能


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib


class LSTMClassifier:
    def __init__(self, classNum, learning_rate=0.001, epochs=100, batch_size=20, hidden_units=(64, 32, 16), is_normalize=True, normalize_type='minmax'):
        """

        :param classNum:  分类个数
        :param hidden_units:  隐藏层神经元个数，tuple类型，例如(64, 32， 16)
        :param learning_rate:  学习率
        :param epochs:  迭代次数
        :param batch_size:  批次大小
        :param is_normalize:  是否归一化
        :param normalize_type:  归一化类型, minmax或者standard
        """
        self.classNum = classNum
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_normalize = is_normalize
        self.normalize_type = normalize_type
        self.model = None

    def __reshape_data(self, data):
        return data.reshape(data.shape[0], data.shape[1], 1)

    def __normalize_data(self, data, scalerPath):
        if self.normalize_type == 'minmax':
            scaler = MinMaxScaler()
        elif self.normalize_type == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError('normalize_type must be minmax or standard')
        data = scaler.fit_transform(data)
        joblib.dump(scaler, scalerPath)
        return data

    def __build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_units[0], return_sequences=True))
        self.model.add(LSTM(self.hidden_units[0]))
        self.model.add(Dense(self.hidden_units[1], activation='selu'))
        self.model.add(Dense(self.hidden_units[2], activation='selu'))
        self.model.add(Dense(self.classNum, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, train_data, train_labels, filepath="./LSTM-Model.h5", scalerPath="./LSTM-Scaler-Model.z", val_data=None, val_labels=None):
        """
        训练模型
        :param train_data:  训练数据
        :param train_labels:  训练标签, one-hot编码
        :param val_data:  验证数据
        :param val_labels:  验证标签, one-hot编码
        :param filepath:  模型保存路径
        :return:
        """
        self.__build_model()
        if self.is_normalize:
            train_data = self.__normalize_data(train_data, scalerPath)
            if val_data is not None and val_labels is not None:
                scaler = joblib.load(scalerPath)
                val_data = scaler.transform(val_data)
        train_data = self.__reshape_data(train_data)
        if val_data is not None and val_labels is not None:
            val_data = self.__reshape_data(val_data)
        if val_data is None and val_labels is None:
            self.model.fit(train_data, train_labels, epochs=self.epochs, batch_size=self.batch_size)
        else:
            self.model.fit(train_data, train_labels, validation_data=(val_data, val_labels),
                           epochs=self.epochs, batch_size=self.batch_size)
        # 保存模型
        self.model.save(filepath)

    def predict(self, test_data, filepath="./LSTM-Model.h5", scalerPath="./LSTM-Scaler-Model.z"):
        """
        预测
        :param test_data:  测试数据
        :param filepath:  模型保存路径
        :return:
        """
        if self.is_normalize:
            scaler = joblib.load(scalerPath)
            test_data = scaler.transform(test_data)
        test_data = self.__reshape_data(test_data)
        model = self.load_model(filepath)
        predict_labels = model.predict(test_data)
        return predict_labels

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        model = tf.keras.models.load_model(filepath)
        return model
