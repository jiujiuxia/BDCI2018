#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import random

# 设置随机数种子
random.seed(42)


# 这个函数通过测试集的Y标签和预测的Y标签来计算精确率、召回率以及F1值并将它们输出，并输出淆矩阵以及训练集和测试集上的误差率
# truth参数给出了测试集中的Y标签
# predicated参数给出了预测的Y标签
def printResult(truth, predicated):
    # 采用加权平均来计算精确率、召回率和F1值并输出
    precision = precision_score(truth.tolist(), predicated.tolist(), average='weighted')
    recall = recall_score(truth, predicated, average='weighted')
    f1 = f1_score(truth, predicated, average='weighted')
    print("Precision", precision)
    print("Recall", recall)
    print("f1_score", f1)
    # 输出混淆矩阵，训练集和测试集的误差率
    print("confusion_matrix")
    print(confusion_matrix(truth, predicated))
    print('predicting, classification error=%f' % (
            sum(int(predicated[i]) != truth[i] for i in range(len(truth))) / float(len(truth))))
    # 返回精确率、召回率以及F1值
    return precision, recall, f1


# 这个函数将给定的文件保存成utf-8格式
# file_path参数给出了要保存文件的路径
# content参数给出了要保存哪些部分的内容
# para_mode参数设置了文件模式，这里默认是二进制读写
def save_utf8(file_path, content, para_mode="wb"):
    """
    save the given content to the given path
    """
    """
    save the given content to the given path
    """
    # 调用open函数来写入要保存的内容
    with open(file_path, para_mode) as log:
        log.write(content)


# 这个函数将在训练集上训练Xgboost模型，并在测试集上作预测来评估模型的效果
# train_X参数要求输入训练集的解释变量X
# train_Y参数要求输入训练集的Y标签
# test_X参数要求输入测试集的解释变量X
# test_Y参数要求输入测试集的Y标签
def xgTestSelfDataset(train_X, train_Y, test_X, test_Y):
    import xgboost as xgb
    import time

    # 将训练数据和测试数据转换成DMatrix对象
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    # setup parameters for xgboost
    param = {
        # 定义学习任务及相应的学习目标
        # 这里选择的是处理多分类问题的目标函数，并输出样本所属于每个类别的概率
        'objective': 'multi:softprob',
        # 学习率
        "learning_rate": 0.05,
        # 树的最大深度，默认值是6，通常取值：3-10
        # 该参数越大，越容易过拟合
        'max_depth': 9,
        # 为了防止过拟合，更新过程中用到的收缩步长
        # 默认值是0.3，取值范围为：[0,1]，通常最后设置eta为0.01~0.2
        'eta': 0.2,
        # 这个参数是最小样本权重的和，用于防止过拟合
        # 当它的值较大时，可以避免模型学习到局部的特殊样本，但是如果这个值过高，会导致欠拟合
        'min_child_weight': 10,
        # silent默认是0，当这个参数值为1时，静默模式开启，不会输出任何信息
        'silent': 1,
        # L1正则的惩罚系数,可以用来降低过拟合，默认值是0
        'alpha': 0.001,
        # L2正则的惩罚系数,可以用来降低过拟合，默认值是0
        'lambda': 0.002,
        # 类的数目
        'num_class': 2,
        # 树的数量
        "n_estimators": 500,
        # 控制对于每棵树，随机采样的比例。减小这个参数的值，可以避免过拟合
        # 典型取值：0.5-1
        "subsample": 0.8,
        # 用来控制在建立树时对特征随机采样的比例，默认值是1
        'colsample_bytree': 0.8,
        # 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
        "scale_pos_weight": 0.75,
        # 随机数的种子，设置它可以复现随机数据的结果，也可以用于调整参数
        "seed": 23}
    # 生成list
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    # 迭代次数
    num_round = 300
    # 记录此刻的时间
    start = time.time()
    # 调用train函数训练模型
    bst = xgb.train(param, xg_train, num_round, watchlist)
    # 计算训练过程所花费的时间
    trainDuration = time.time() - start
    start = time.time()
    # 调用predict函数作预测
    yprob = bst.predict(xg_test).reshape(test_Y.shape[0], 11)
    # 计算测试过程所花费的时间
    testDuration = time.time() - start
    # 根据预测的概率给Y赋0,1标签
    ylabel = np.argmax(yprob, axis=1)
    # 保存训练好的模型
    if os.path.exists("rhar.model"):
        os.remove("rhar.model")
    bst.save_model("rhar.model")
    # 将测试集的标签和解释变量分别提取到list中
    l_test_x = test_X.tolist()
    l_test_y = test_Y.tolist()
    # 保存预测的结果
    if os.path.exists("rhar.test"):
        os.remove("rhar.test")
    # 调用save_utf8函数将预测的结果转存成utf8格式
    for i in range(0, len(l_test_y)):
        line = "{},".format(l_test_y[i])
        for j in range(0, len(l_test_x[i])):
            line = line + "{},".format(l_test_x[i][j])
        save_utf8("rhar.test", line[0: len(line) - 1] + "\n", "a")
    # 调用printResult函数输出结果并返回
    print(test_Y.shape)
    print(ylabel.shape)
    return printResult(test_Y, ylabel), trainDuration, testDuration, "XGBoost"


# 这个函数主要用于数据的导入以及训练集和测试集的划分，并且主要主函数来调用其他函数
def test_self_dataset():
    # 使用read_csv函数来读入处理好的csv文件，要求输入文件的路径以及分隔符
    total_dataset = pd.read_csv("xi.csv")
    print(total_dataset.head())
    # 按照7:3的比例划分训练集和测试集
    train_test_split = np.random.rand(len(total_dataset)) < 0.75
    train = np.asmatrix(total_dataset[train_test_split])
    test = np.asmatrix(total_dataset[~train_test_split])
    # 由于第1列是id，对分类不起作用，故去掉
    train_x = train[:, 1:train.shape[1] - 1]
    train_y = train[:, train.shape[1] - 1]
    test_x = test[:, 1:train.shape[1] - 1]
    test_y = test[:, train.shape[1] - 1]
    print("train.shape = {}, test_x = {}".format(train_x.shape, test_x.shape))
    # 调用xgTestSelfDataset函数来建模
    xgTestSelfDataset(train_x, train_y, test_x, test_y)


if __name__ == "__main__":
    test_self_dataset()
