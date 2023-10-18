#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :SensorFaultDetect.py
# @Time      :2023/10/2 16:17
# @Author    :LongFei Shan
from abc import ABC, abstractmethod
import pickle
import numpy as np


class SensorFaultDetect(ABC):
    @abstractmethod
    def fit(self):
        """anomaly detection model training
        """
        pass

    @abstractmethod
    def predict(self):
        """anomaly detection model prediction
        """
        pass
    
    @abstractmethod
    def threshold(self):
        """calculate threshold
        """
        pass

    @abstractmethod
    def sensor_fault_location_predict(self):
        """sensor fault location model prediction
        """
        pass
    
    @abstractmethod
    def sensor_fault_location_threshold(self):
        """calculate threshold
        """
        pass
    
    def _box_plot(self, result, is_zero=False):
        # 使用箱线图计算上下限阈值
        up_threshold = []
        down_threshold = []
        for i in range(result.shape[1]):
            q1 = np.percentile(result[:, i], 25)
            q3 = np.percentile(result[:, i], 75)
            iqr = q3 - q1
            up = q3 + 1.5 * iqr
            down = q1 - 1.5 * iqr
            up_threshold.append(up)
            if is_zero:
                if down < 0:
                    down_threshold.append(0)
                else:
                    down_threshold.append(down)
            else:
                down_threshold.append(down)
        return up_threshold, down_threshold

    def save_model(self, file_save_path):
        """save model

        Args:
            file_save_path (string): save path
        """
        with open(file_save_path, "wb") as file:
            pickle.dump((SensorFaultDetect, self), file)

    @staticmethod
    def load_model(file_save_path):
        """load model

        Args:
            file_save_path (str): load path

        Returns:
            _type_: _description_
        """
        with open(file_save_path, "rb") as file:
            MyClass, obj = pickle.load(file)
        return obj



if __name__ == "__main__":
    pass
