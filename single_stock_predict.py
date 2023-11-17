# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:22:30 2023

@author: XF
"""



import torch
import torch.nn as nn#模块包含了许多神经网络的基本构建块，如卷积层、线性层、激活函数等，以及一些高级的层，如自编码器、变分自编码器、循环神经网络等。此外，它还提供了一些用于损失函数和优化算法的类。
import torch.optim as optim#它提供了各种用于优化神经网络的算法。它是一个实现各种优化算法的包，其中包括最常用的优化方法，如随机梯度下降（SGD）、Adam、RMSProp 等
import torch.nn.functional as F#它提供了许多用于处理张量（tensor）的实用函数。这些函数包括各种激活函数（如 ReLU、Sigmoid、Tanh 等）、损失函数（如交叉熵损失、均方误差损失等）、池化函数（如最大池化、平均池化等）以及其他一些常用操作。
from torchvision import datasets, transforms#torchvision 是一个与 PyTorch 配合使用的数据处理和图像转换工具包，它提供了许多预先定义好的数据转换操作，用于将图像数据转换为适合神经网络模型训练的格式。datasets.transforms 模块中包含了许多图像转换类，例如随机裁剪、缩放、旋转、翻转等。这些转换类可以方便地用于图像预处理，例如将图像调整为模型所需的大小、应用数据增强技术等。
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from single_stock_train import CustomDataset, CNN




# Assuming you have a model stored in a file named 'model.pth'
model = CNN()
state_dict = torch.load('D:/FuXuan/Coding/单指数/单指数训练训练.pth')

# 会出现加载错误，还没有解决
model.load_state_dict(state_dict,strict=False)

# Assuming you have a data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Iterate over the data
for data, target in data_loader:
    # Forward pass
    output = model(data)
    # Get the index of the max log-probability
    pred = output.argmax(dim=1, keepdim=True)
    # Print the prediction
    print(pred)