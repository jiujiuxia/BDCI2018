import pandas as pd
import numpy as np

# 使用read_csv函数来读入处理好的csv文件，要求输入文件的路径以及分隔符
total_dataset = pd.read_csv("train1010_1new.csv")
print(total_dataset.head())

# 按照7:3的比例划分训练集和测试集
train_test_split = np.random.rand(len(total_dataset)) < 0.8
train = total_dataset[train_test_split]
test = total_dataset[~train_test_split]

print(train.head(5))
print(test.head(5))
train.to_csv('train_1.csv')
test.to_csv('test_1.csv')