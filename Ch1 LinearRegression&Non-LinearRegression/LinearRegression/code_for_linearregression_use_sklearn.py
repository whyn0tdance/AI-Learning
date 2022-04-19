#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName     :code_for_linearregression_use_sklearn.py
# @Project      :AI-Learning
# @IDE          :PyCharm
# @Time         :2022/4/18 21:33
# @Author       :Frank Yang
# @E-Mail       :whynotdance.franky@gmail.com
"""
根据唐博士讲课以及sklearn工具包使用线性回归处理、预测数据，一维特征输入
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == '__main__':
    # 使用本地幸福报告数据路径
    dataset = pd.read_csv('./data/world-happiness-report-2017.csv')

    # 得到训练与测试数据
    train_data = dataset.sample(frac=0.8)  # 训练数据，使用sample函数按80%的比例随即抽样出数据
    test_data = dataset.drop(train_data.index)  # 测试数据，使用余下的比例的数据

    # 确定使用数据集中哪一列作为输入特征及其标签
    input_feature_name = 'Economy..GDP.per.Capita.'  # 输入特征
    output_lable_name = 'Happiness.Score'  # 输出特征（标签）

    x_train = train_data[input_feature_name].values
    y_train = train_data[output_lable_name].values

    x_test = test_data[input_feature_name].values
    y_test = test_data[output_lable_name].values

    # 将训练集、测试集数据转化为2维矩阵（数组）
    x_train = x_train.reshape(x_train.shape[0], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # 实例化模型
    model = linear_model.LinearRegression()

    # 画出训练集与测试集样本
    plt.scatter(x_train, y_train, label='Train data')
    plt.scatter(x_test, y_test, label='Test data')
    plt.xlabel(input_feature_name)
    plt.ylabel(output_lable_name)
    plt.title('Relationship between ' + input_feature_name + ' ' + output_lable_name)
    plt.legend()
    plt.show()

    # 使用训练集训练模型
    model.fit(x_train, y_train)

    # 使用测试集测试数据
    y_predicted = model.predict(x_test)

    # 得到训练模型的估计系数
    Coefficients_train = model.coef_

    # 打印系数
    print('训练估计系数为: {}'.format(Coefficients_train))
    # 打印训练模型的平均误差
    print('均方误差回归损失: {}'.format(mean_squared_error(y_test, y_predicted)))
    # The coefficient of determination: 1 is perfect prediction
    print('R^2（确定系数）回归得分函数（1为）: {}'.format(r2_score(y_test, y_predicted)))

    # 得到用于画图的预测数据
    predictions_num = 100
    x_predicted_for_plot = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
    y_predicted_for_plot = model.predict(x_predicted_for_plot)

    # 画出练集与测试集样本以及预测模型
    plt.scatter(x_train, y_train, label='Train data')
    plt.scatter(x_test, y_test, label='Test data')
    plt.plot(x_predicted_for_plot, y_predicted_for_plot, 'r', label='Prediction')
    plt.xlabel(input_feature_name)
    plt.ylabel(output_lable_name)
    plt.title('Relationship between ' + input_feature_name + ' ' + output_lable_name)
    plt.legend()
    plt.show()


