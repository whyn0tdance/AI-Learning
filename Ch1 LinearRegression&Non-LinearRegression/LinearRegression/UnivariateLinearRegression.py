import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('./data/world-happiness-report-2017.csv')

# 得到训练和测试数据
train_data = data.sample(frac=0.8)  # 训练数据，使用sample函数按80%的比例随即抽样出数据
test_data = data.drop(train_data.index)  # 测试数据，使用余下的比例的数据

input_param_name = 'Economy..GDP.per.Capita.'  # 输入特征
output_param_name = 'Happiness.Score'  # 输出特征（标签）

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

"""
画出训练集与测试集样本
"""
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

num_iterations = 5000  # 迭代次数
learning_rate = 0.001  # 学习率

linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

"""
画出迭代次数与损失
"""
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()
