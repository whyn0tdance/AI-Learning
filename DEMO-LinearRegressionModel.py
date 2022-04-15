#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName     :test.py
# @Project      :AI-Learning
# @IDE          :PyCharm
# @Time         :2022/4/14 19:46
# @Author       :Frank Yang
# @E-Mail       :whynotdance.franky@gmail.com

import torch
import torch.nn as nn
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

if __name__ == '__main__':
    x_values = [i for i in range(11)]   # 输入数据x
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i + 1 for i in x_values]  # 对应标签y
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    input_dim = 1   # 输入维度
    output_dim = 1  # 输出维度

    model = LinearRegressionModel(input_dim, output_dim)

    # 使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 10000  # 训练次数
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # 优化器SGD
    criterion = nn.MSELoss()    # 损失函数MSE

    # 训练模型
    for epoch in range(epochs):
        epoch += 1

        # np.array转换成tensor——CPU
        # inputs = torch.from_numpy(x_train)
        # labels = torch.from_numpy(y_train)
        # np.array转换成tensor——GPU
        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)

        # 每一次迭代梯度要清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新权重参数
        optimizer.step()

        if epoch % 50 ==0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    # 预测模型 预测结果
    # predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()

    # 模型的保存与读取
    torch.save(model.state_dict(), 'model.pkl')
    # model.load_state_dict(torch.load('model.pkl'))
