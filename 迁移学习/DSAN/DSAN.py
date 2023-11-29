#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :DDC.py
# @Time      :2023/11/26 15:54
# @Author    :LongFei Shan
import torch
from torch import nn
from Pytorch学习.LMMD import LMMD_loss
from visdom import Visdom
import numpy as np
from Pytorch学习.MMD import MMDLoss, MMDLoss_


class DSAN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(1, 5), stride=(1, 1)),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=0)).to(device)
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(1, 3), stride=(1, 1)),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=0)).to(device)
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(1, 3), stride=(1, 1)),
                                    nn.ReLU()).to(device)
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=(1, 1)),
                                    nn.ReLU()).to(device)
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(1, 3), stride=(1, 1)),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=0)).to(device)
        self.layer6 = nn.Sequential(nn.Flatten()).to(device)
        self.layer7 = nn.Sequential(nn.Linear(in_features=1280, out_features=4096), nn.ReLU(),
                                    nn.Dropout(p=0.3)).to(device)
        self.layer8 = nn.Sequential(nn.Linear(in_features=4096, out_features=4096), nn.ReLU(), nn.Dropout(p=0.3)).to(device)
        self.layer9 = nn.Sequential(nn.Linear(in_features=4096, out_features=10)).to(device)

    def forward(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs = torch.from_numpy(inputs).type(torch.float32).to(device)
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        upper_output = self.layer8(x)
        output = self.layer9(upper_output)
        return output, upper_output

class DSAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ddc_model = DSAN_Model()

    def forward(self, source_inputs, target_inputs):
        s_output, s_upper_output = self.ddc_model(source_inputs)
        t_output, t_upper_output = self.ddc_model(target_inputs)
        return s_output, t_output, s_upper_output, t_upper_output


def train(num_class, source_data, target_data, source_label, target_label, source_vec_label, target_vec_label, epochs=100, batch_size=100):
    # TODO: train
    ddc = DSAN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=ddc.parameters(), lr=0.001)
    # 检查每个参数的 requires_grad 属性
    for name, param in ddc.named_parameters():
        print(f'Parameter: {name}, Requires Grad: {param.requires_grad}')
    lmmd_loss = LMMD_loss(class_num=num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None)
    # 使用MMD损失函数进行训练
    viz = Visdom(env="main")
    viz.line([[0, 0, 0]], [0], win="DDC_train_Loss", opts=dict(title="DDC train Loss", legend=["loss", "acc_s", "acc_t"]))
    for epoch in range(epochs):
        for i in range(source_data.shape[0]//batch_size):
            optimizer.zero_grad()
            # 从target_data中随机挑选batch_size个数据
            source_data_temp = source_data[i*batch_size:(i+1)*batch_size]
            source_label_temp = source_label[i*batch_size:(i+1)*batch_size].reshape(-1, 1)
            source_vec_label_temp = source_vec_label[i*batch_size:(i+1)*batch_size]
            if i >= target_data.shape[0]//batch_size:
                j = i % (target_data.shape[0]//batch_size)
            else:
                j = i
            target_data_temp = target_data[j * batch_size:(j + 1) * batch_size]
            target_label_temp = target_label[j * batch_size:(j + 1) * batch_size].reshape(-1, 1)
            target_vec_label_temp = target_vec_label[j*batch_size:(j+1)*batch_size]
            # 训练
            source_outputs, target_outputs, s_x, t_x = ddc(source_data_temp, target_data_temp)
            # 计算损失
            loss = lmmd_loss(s_x, t_x, source_label_temp, target_label_temp, True) + loss_func(source_outputs, torch.from_numpy(source_vec_label_temp).to(device))
            loss.backward()
            optimizer.step()
        source_outputs, target_outputs, _, _ = ddc(source_data, target_data)
        # 计算准确率，源域
        source_outputs = source_outputs.cpu().detach().numpy()
        source_outputs = np.argmax(source_outputs, axis=1)
        acc_s = np.mean(source_outputs == source_label)
        # 计算准确率，目标域
        target_outputs = target_outputs.cpu().detach().numpy()
        target_outputs = np.argmax(target_outputs, axis=1)
        acc_t = np.mean(target_outputs == target_label)
        viz.line([[loss.item(), acc_s, acc_t]], [epoch], update="append", opts=dict(title="DDC train Loss", legend=["loss", "acc_s", "acc_t"]), win="DDC_train_Loss")

        print(f"epoch:{epoch}, loss:{loss.item()}, acc_s:{acc_s}, acc_t:{acc_t}")

