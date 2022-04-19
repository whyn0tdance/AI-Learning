"""""""""""""""""""""""""""""""""""""""""""""""""""
!/usr/bin/env python
 -*- coding:utf-8 -*-
 @FileName     :code_for_MultivariateLinearRegression_use_sklearn.py
 @Project      :AI-Learning
 @IDE          :PyCharm
 @Time         :2022/4/18 22:41
 @Author       :Frank Yang
 @E-Mail       :whynotdance.franky@gmail.com
 根据唐博士讲课以及sklearn工具包使用线性回归处理、预测数据，多维特征输入线性回归
"""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score




if __name__ == '__main__':
    # 使用的数据路径
    dataset = pd.read_csv('./data/world-happiness-report-2017.csv')

    # 得到训练与测试数据
    train_data = dataset.sample(frac=0.8)  # 训练数据，使用sample函数按80%的比例随即抽样出数据
    test_data = dataset.drop(train_data.index)  # 测试数据，使用余下的比例的数据

    # 确定使用数据集中哪一列作为输入特征及其标签
    input_feature_name_1 = 'Economy..GDP.per.Capita.'  # 输入特征1
    input_feature_name_2 = 'Freedom'    # 输入特征2
    output_lable_name = 'Happiness.Score'  # 输出特征（标签）

    x_train = train_data[[input_feature_name_1, input_feature_name_2]].values
    y_train = train_data[[output_lable_name]].values

    x_test = test_data[[input_feature_name_1, input_feature_name_2]].values
    y_test = test_data[[output_lable_name]].values

    # 实例化模型
    model = linear_model.LinearRegression()

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
    # 打印得分，1为满分
    print('R^2（确定系数）回归得分函数（1为）: {}'.format(r2_score(y_test, y_predicted)))

    # 画出训练集与测试集样本
    plot_training_trace = go.Scatter3d(
        x=x_train[:, 0].flatten(),
        y=x_train[:, 1].flatten(),
        z=y_train.flatten(),
        name='Training Set',
        mode='markers',
        marker={
            'size': 10,
            'opacity': 1,
            'line': {
                'color': 'rgb(255, 255, 255)',
                'width': 1
            },
        }
    )

    plot_test_trace = go.Scatter3d(
        x=x_test[:, 0].flatten(),
        y=x_test[:, 1].flatten(),
        z=y_test.flatten(),
        name='Test Set',
        mode='markers',
        marker={
            'size': 10,
            'opacity': 1,
            'line': {
                'color': 'rgb(255, 255, 255)',
                'width': 1
            },
        }
    )

    plot_layout = go.Layout(
        title='Date Sets',
        scene={
            'xaxis': {'title': input_feature_name_1},
            'yaxis': {'title': input_feature_name_2},
            'zaxis': {'title': output_lable_name}
        },
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    # 得到用于画图的预测数据
    predictions_num = 10

    x_min = x_train[:, 0].min()
    x_max = x_train[:, 0].max()

    y_min = x_train[:, 1].min()
    y_max = x_train[:, 1].max()

    x_axis = np.linspace(x_min, x_max, predictions_num)
    y_axis = np.linspace(y_min, y_max, predictions_num)

    x_predictions = np.zeros((predictions_num * predictions_num, 1))
    y_predictions = np.zeros((predictions_num * predictions_num, 1))

    x_y_index = 0
    for x_index, x_value in enumerate(x_axis):
        for y_index, y_value in enumerate(y_axis):
            x_predictions[x_y_index] = x_value
            y_predictions[x_y_index] = y_value
            x_y_index += 1

    z_predictions = model.predict(np.hstack((x_predictions, y_predictions)))

    plot_predictions_trace = go.Scatter3d(
        x=x_predictions.flatten(),
        y=y_predictions.flatten(),
        z=z_predictions.flatten(),
        name='Prediction Plane',
        mode='markers',
        marker={
            'size': 1,
        },
        opacity=0.8,
        surfaceaxis=2,
    )

    # 画出练集与测试集样本以及预测模型
    plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
    plot_figure = go.Figure(data=plot_data, layout=plot_layout)
    plotly.offline.plot(plot_figure)