#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MMD.py
# @Time      :2023/11/28 10:41
# @Author    :LongFei Shan
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

# 定义MMD损失函数
class MMDLoss_(nn.Module):
    # 网上嫖的代码
    def __init__(self, kernel_type='gaussian', kernel_mul=8.0, kernel_num=5):
        super(MMDLoss_, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = None
        self.use_cuda = torch.cuda.is_available()

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        if self.use_cuda:
            source = source.cuda()
            target = target.cuda()

        batch_size = int(source.size()[0])

        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])

        loss = XX + YY - XY - YX
        loss = torch.clamp(loss, min=0.0)  # 确保损失为非负值

        return loss


class MMDLoss(nn.Module):
    # 重新写的代码
    def __init__(self, kernel_type='gaussian', kernel_mul=8.0, kernel_num=5, fix_sigma=None, degree=3):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.degree = degree
        self.use_cuda = torch.cuda.is_available()

    def gaussian_kernel(self, X, Y, sigma):
        """
        计算高斯核函数
        X: Tensor，形状为 (batch_size, feature_dim)
        Y: Tensor，形状为 (batch_size, feature_dim)
        sigma: 高斯核函数的标准差
        返回形状为 (batch_size, batch_size) 的核矩阵
        """
        M = X.size(0)
        N = Y.size(0)
        X = X.unsqueeze(1).expand(-1, N, -1)
        Y = Y.unsqueeze(0).expand(M, -1, -1)
        L2_distance = ((X - Y) ** 2).sum(2)
        n_samples = int(X.size()[0])
        if sigma is None:
            sigma = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples)
        sigma /= self.kernel_mul ** (self.kernel_num // 2)
        sigma_list = [sigma * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / (sigma_temp + 1e-5)) for sigma_temp in sigma_list]
        kernel_matrix = sum(kernel_val)
        #print(f"kernel_matrix: {kernel_matrix}")
        return kernel_matrix


    def linear_kernel(self, X, Y):
        """
        计算线性核函数
        X: Tensor，形状为 (batch_size, feature_dim)
        Y: Tensor，形状为 (batch_size, feature_dim)
        返回形状为 (batch_size, batch_size) 的核矩阵
        """
        return torch.matmul(X, Y.t())


    def cosine_kernel(self, X, Y):
        """
        计算余弦核函数
        X: Tensor，形状为 (batch_size, feature_dim)
        Y: Tensor，形状为 (batch_size, feature_dim)
        返回形状为 (batch_size, batch_size) 的核矩阵
        """
        X_norm = F.normalize(X, dim=1)
        Y_norm = F.normalize(Y, dim=1)
        return torch.matmul(X_norm, Y_norm.t())


    def polynomial_kernel(self, X, Y, degree):
        """
        计算多项式核函数
        X: Tensor，形状为 (batch_size, feature_dim)
        Y: Tensor，形状为 (batch_size, feature_dim)
        degree: 多项式核函数的次数
        返回形状为 (batch_size, batch_size) 的核矩阵
        """
        return (torch.matmul(X, Y.t()) + 1) ** degree


    def forward(self, X, Y):
        """
        计算最大均值差异（MMD）损失函数
        X: Tensor，形状为 (batch_size, feature_dim)
        Y: Tensor，形状为 (batch_size, feature_dim)
        kernel_type: 核函数类型，可选值为 'gaussian', 'linear', 'cosine', 'polynomial'
        kernel_param: 核函数的参数，如高斯核函数的标准差、多项式核函数的次数等
        返回 MMD 损失
        """
        batch_size = X.size(0)
        K_xx, K_yy, K_xy, K_yx = 0, 0, 0, 0
        if self.kernel_type == 'gaussian':
            K_xx = self.gaussian_kernel(X, X, self.fix_sigma)
            K_yy = self.gaussian_kernel(Y, Y, self.fix_sigma)
            K_xy = self.gaussian_kernel(X, Y, self.fix_sigma)
            K_yx = self.gaussian_kernel(Y, X, self.fix_sigma)
        elif self.kernel_type == 'linear':
            K_xx = self.linear_kernel(X, X)
            K_yy = self.linear_kernel(Y, Y)
            K_xy = self.linear_kernel(X, Y)
            K_yx = self.linear_kernel(Y, X)
        elif self.kernel_type == 'cosine':
            K_xx = self.cosine_kernel(X, X)
            K_yy = self.cosine_kernel(Y, Y)
            K_xy = self.cosine_kernel(X, Y)
            K_yx = self.cosine_kernel(Y, X)
        elif self.kernel_type == 'poly':
            K_xx = self.polynomial_kernel(X, X, self.degree)
            K_yy = self.polynomial_kernel(Y, Y, self.degree)
            K_xy = self.polynomial_kernel(X, Y, self.degree)
            K_yx = self.polynomial_kernel(Y, X, self.degree)

        loss = torch.mean(K_xx) + torch.mean(K_yy) - torch.mean(K_xy) - torch.mean(K_yx)
        # print(f"loss: {loss}")
        loss = torch.clamp(loss, min=0.0)

        return loss
