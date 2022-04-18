"""Normalize features"""

import numpy as np


def normalize(features):
    """
    标准化数据=（原数据-数据均值）/标准差
    :param features: m*n矩阵
    :return: features_normalized：标准化后的数据, features_mean：数据均值, features_deviation：数据标准差
    """
    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
