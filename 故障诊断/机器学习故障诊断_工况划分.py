#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :机器学习故障诊断.py
# @Time      :2023/8/8 17:57
# @Author    :LongFei Shan
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
# 混淆矩阵
from sklearn.metrics import confusion_matrix
# 画出混淆矩阵
import matplotlib.pyplot as plt
# 分类报告
from sklearn.metrics import classification_report


class MLClassifier:
    def __init__(self, modelName="random_forest", is_normalize=True, normalize_type='minmax'):
        self.modelName = modelName
        self.is_normalize = is_normalize
        self.normalize_type = normalize_type
        self.model = None
        self.__builtModel()

    def __builtModel(self):
        if self.modelName == 'svm':
            self.model = svm.SVC(C=1.0,
                                kernel="rbf",
                                degree=3,
                                gamma="scale",
                                coef0=0.0,
                                shrinking=True,
                                probability=False,
                                tol=1e-3,
                                cache_size=200,
                                class_weight=None,
                                verbose=False,
                                max_iter=-1,
                                decision_function_shape="ovr",
                                break_ties=False,
                                random_state=None)
        elif self.modelName == 'knn':
            self.model = KNeighborsClassifier(n_neighbors=5,
                                                weights="uniform",
                                                algorithm="auto",
                                                leaf_size=30,
                                                p=2,
                                                metric="minkowski",
                                                metric_params=None,
                                                n_jobs=None)
        elif self.modelName == 'decision_tree':
            self.model = DecisionTreeClassifier(criterion="gini",
                                                splitter="best",
                                                max_depth=None,
                                                min_samples_split=2,
                                                min_samples_leaf=1,
                                                min_weight_fraction_leaf=0.0,
                                                max_features=None,
                                                random_state=None,
                                                max_leaf_nodes=None,
                                                min_impurity_decrease=0.0,
                                                class_weight=None,
                                                ccp_alpha=0.0)
        elif self.modelName == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100,
                                                criterion="gini",
                                                max_depth=None,
                                                min_samples_split=2,
                                                min_samples_leaf=1,
                                                min_weight_fraction_leaf=0.0,
                                                max_features="sqrt",
                                                max_leaf_nodes=None,
                                                min_impurity_decrease=0.0,
                                                bootstrap=True,
                                                oob_score=False,
                                                n_jobs=None,
                                                random_state=None,
                                                verbose=0,
                                                warm_start=False,
                                                class_weight=None,
                                                ccp_alpha=0.0,
                                                max_samples=None)
        elif self.modelName == 'logistic_regression':
            self.model = LogisticRegression(penalty="l2",
                                            dual=False,
                                            tol=1e-4,
                                            C=1.0,
                                            fit_intercept=True,
                                            intercept_scaling=1,
                                            class_weight=None,
                                            random_state=None,
                                            solver="lbfgs",
                                            max_iter=100,
                                            multi_class="auto",
                                            verbose=0,
                                            warm_start=False,
                                            n_jobs=None,
                                            l1_ratio=None)
        elif self.modelName == 'naive_bayes':
            self.model = GaussianNB()
        else:
            raise ValueError('modelName must be svm, knn, decision_tree, random_forest, logistic_regression, naive_bayes')

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

    def fit(self, train_x, train_y, scalerPath="./ML-Scaler-Model.z", filepath="./ML-Model.z"):
        if self.is_normalize:
            train_x = self.__normalize_data(train_x, scalerPath)
        self.model.fit(train_x, train_y)
        joblib.dump(self.model, filepath)

    def predict(self, test_x, scalerPath="./ML-Scaler-Model.z", filepath="./ML-Model.z"):
        if self.is_normalize:
            scaler = joblib.load(scalerPath)
            test_x = scaler.transform(test_x)
        model = joblib.load(filepath)
        return model.predict(test_x)

    def print_report(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))

    def print_confusion_matrix(self, y_test, y_pred):
        # 打印百分比
        print(confusion_matrix(y_test, y_pred, normalize='true'))

