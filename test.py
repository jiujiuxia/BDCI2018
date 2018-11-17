# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 14:02:21 2018

@author: jiuxia
"""
import pandas as pd
import xgboost as xgb
import numpy as np

test_dataset=pd.read_csv("test1013_4.0.csv")
print(test_dataset.head())
test_dataset= np.asmatrix(test_dataset)


test_x = test_dataset[:, 1:test_dataset.shape[1]]
test_y = test_dataset[:, test_dataset.shape[1] - 1]
print("test_x = {}".format(test_x.shape))

xg_test = xgb.DMatrix(test_x, label=test_y)

#调用xgTestSelfDataset函数来建模
tar = xgb.Booster(model_file='rhar.model')

preds = tar.predict(xg_test).reshape(test_y.shape[0],11)
ylabel = np.argmax(preds, axis=1)
print('start output')
yfinal = pd.DataFrame(ylabel)
yfinal.to_excel('final_1013_4.0.xlsx')
