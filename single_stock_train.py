# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:22:30 2023

@author: faiz
"""


import torch
import torch.nn as nn#模块包含了许多神经网络的基本构建块，如卷积层、线性层、激活函数等，以及一些高级的层，如自编码器、变分自编码器、循环神经网络等。此外，它还提供了一些用于损失函数和优化算法的类。
import torch.optim as optim#它提供了各种用于优化神经网络的算法。它是一个实现各种优化算法的包，其中包括最常用的优化方法，如随机梯度下降（SGD）、Adam、RMSProp 等
import torch.nn.functional as F#它提供了许多用于处理张量（tensor）的实用函数。这些函数包括各种激活函数（如 ReLU、Sigmoid、Tanh 等）、损失函数（如交叉熵损失、均方误差损失等）、池化函数（如最大池化、平均池化等）以及其他一些常用操作。
from torchvision import datasets, transforms#torchvision 是一个与 PyTorch 配合使用的数据处理和图像转换工具包，它提供了许多预先定义好的数据转换操作，用于将图像数据转换为适合神经网络模型训练的格式。datasets.transforms 模块中包含了许多图像转换类，例如随机裁剪、缩放、旋转、翻转等。这些转换类可以方便地用于图像预处理，例如将图像调整为模型所需的大小、应用数据增强技术等。
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score
#使用leakyrelu只取0和最大值，锐化了卷积核过滤后的输出
#定义网络模型：
class CNN(nn.Module):
    def __init__(self):#定义
        super(CNN, self).__init__()#?
        self.conv1 = nn.Conv2d(in_channels=1,out_channels = 64,kernel_size=(5,3),stride=(1,3),dilation=(1,2))
        self.conv2 = nn.Conv2d(64,128,kernel_size=(5,3),stride=(1,3),dilation=(1,2))
        self.conv3 = nn.Conv2d(128,256,kernel_size=(5,3),stride=(1,3),dilation=(1,2))
        self.fc1 = nn.Linear(69888,2)


    def forward(self, x):#前向传播
        x = self.conv1(x)
        x = nn.LeakyReLU(0.1)(x)#leakyReLU 函数进行非线性变换更鲁棒性
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        
        x = self.conv2(x)
        x = nn.LeakyReLU(0.1)(x)
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        
        x = self.conv3(x)
        x = nn.LeakyReLU(0.1)(x)
        x = nn.MaxPool2d(kernel_size=(2, 1))(x)
        
        #x = self.dropout1(x)#防止过拟合和减小模型的方差。
        x = torch.flatten(x, 1)#将特征图展平为一维向量
        x = self.fc1(x)#展平后的数据传递给全连接层 fc1
        output = nn.LogSoftmax(dim=1)(x)#LogSoftmax 函数将输出转换为概率分布，以表示手写数字
        return output
    
#定义训练函数：
def train(model, train_loader, optimizer, criterion, device):#用于训练深度学习模型
    model.train()#将模型设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):#获取每个批次的输入数据（data）和目标标签（target）
        data, target = data.to(device), target.to(device)#将数据和标签移动到指定的设备（gpu）
        optimizer.zero_grad()#重置优化器的梯度
        output = model(data)#调用模型的 forward 方法，计算模型的输出
        loss = criterion(output, target)#使用损失函数计算预测输出与真实标签之间的差距
        loss.backward()#对损失函数进行反向传播，更新模型参数
        optimizer.step()#执行优化器的一步更新

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.NLLLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Collect predictions and targets for evaluation
            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    auc = roc_auc_score(targets, predictions)
    recall = recall_score(targets, predictions)
    precision = precision_score(targets, predictions)
    positive_mean = np.mean(predictions)
    negative_mean = np.mean(1 - np.array(predictions))
    positive_median = np.median(predictions)
    negative_median = np.median(1 - np.array(predictions))
    
    return test_loss, accuracy, auc, recall, precision, positive_mean, negative_mean, positive_median, negative_median


import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_name = img_name.replace(".png", ".txt")
        label_path = os.path.join(self.label_dir, label_name)
        
        
        image = Image.open(img_path).convert("L")  #转为灰度图
        width, height = image.size
        # 计算裁剪区
        left = width // 9
        top = height // 10
        right = width - left
        bottom = height - top
        # 裁剪图像
        image = image.crop((left, top, right, bottom))
        target_resolution = (400,200)
        # 调整图片大小
        image = image.resize(target_resolution)
        
        
        label = self.read_label_file(label_path)
        label = torch.tensor(int(label))  # 将标签转化为张量
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def read_label_file(self, label_path):
        with open(label_path, "r") as file:
            label = file.read().strip()
        return label
    def compute_mean_std(self):
        all_images = []
        for img_name in self.image_filenames:
            img_path = os.path.join(self.image_dir, img_name)
            image = Image.open(img_path).convert("L")
            all_images.append(np.array(image).flatten())

        all_images = np.stack(all_images)
        mean = np.mean(all_images)
        std = np.std(all_images)

        return mean, std

#对数据进行归一化
transform = transforms.Compose([
    transforms.ToTensor()]) 


image_dir = r'D:/FuXuan/Coding/单指数/jd/images'
label_dir = r'D:/FuXuan/Coding/单指数/jd/labels'


dataset = CustomDataset(image_dir, label_dir, transform)



from torch.utils.data import random_split



# 计算训练集和测试集的大小
total_size = len(dataset)
train_size = int(0.7 * total_size)
test_size = total_size - train_size

# 随机分配训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建训练集和测试集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


#初始化模型、优化器和损失函数：
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)#初始化一个优化器，用于在训练过程中更新模型的参数。这里使用了自适应学习率的优化算法Adam。
criterion = nn.CrossEntropyLoss()#交叉熵损失函数，用于衡量模型预测与实际值之间的差距。用于衡量分类问题中的预测准确性。



# 进行训练和测试
epochs = 10
test_losses = []
accuracies = []
auc_scores = []
recalls = []
precisions = []
positive_means = []
negative_means = []
positive_medians = []
negative_medians = []

for epoch in range(epochs):
    train(model, train_loader, optimizer, criterion, device)
    test_loss, accuracy, auc, recall, precision, pos_mean, neg_mean, pos_median, neg_median = test(model, test_loader, device)
    
    test_losses.append(test_loss)
    accuracies.append(accuracy)
    auc_scores.append(auc)
    recalls.append(recall)
    precisions.append(precision)
    positive_means.append(pos_mean)
    negative_means.append(neg_mean)
    positive_medians.append(pos_median)
    negative_medians.append(neg_median)
    
    print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy*100:.2f}%, AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

save_path = r'D:/FuXuan/Coding/单指数/单指数训练训练.pth'
torch.save(model, save_path)
# 将结果存储在变量中
test_loss_values = np.array(test_losses)
accuracy_values = np.array(accuracies)
auc_scores_values = np.array(auc_scores)
recall_values = np.array(recalls)
precision_values = np.array(precisions)
positive_mean_values = np.array(positive_means)
negative_mean_values = np.array(negative_means)
positive_median_values = np.array(positive_medians)
negative_median_values = np.array(negative_medians)

import pandas as pd
results = pd.DataFrame({
    'Test Loss': test_loss_values,
    'Accuracy': accuracy_values,
    'AUC Score': auc_scores_values,
    'Recall': recall_values,
    'Precision': precision_values,
    'Positive Mean': positive_mean_values,
    'Negative Mean': negative_mean_values,
    'Positive Median': positive_median_values,
    'Negative Median': negative_median_values
})


# 用于输出预测结果
image_dir_pred = r'D:/FuXuan/Coding/单指数/jd/images_pred'
dataset_pred = CustomDataset(image_dir_pred,label_dir,transform)

# Assuming you have a data loader
data_loader = DataLoader(dataset_pred, batch_size=32, shuffle=False)
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

output = pred.numpy()
output_df = pd.DataFrame(output)
print(output_df)
output_df.to_csv(r'D:/FuXuan/Coding/单指数/jd/Pred_output.csv')