# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:28:04 2018

@author: jiuxia
"""

import pandas as pd
import random

random.seed(8888)

data1 = pd.read_csv('yangkang0.74410009000.csv')
data2 = pd.read_csv('lgb_15.0(73.94).csv')
data3 = pd.read_csv('lgb_16.0(74.19).csv')

data = []

for i in range(len(data1)):
    each = []
    each.append(data1.iloc[i, 1])
    each.append(data2.iloc[i, 1])
    each.append(data3.iloc[i, 1])
    # print(each)
    a = {}
    for j in each:
        if each.count(j) > 1:
            a[j] = each.count(j)
    if a:
        a = sorted(a.items(), key=lambda item: item[0])
        # print (a[0][0])
        data.append(a[0][0])
    else:
        r = random.uniform(0, 1)
        print(r)
        print(each)
        if r <= 0.35:
            data.append(data1.iloc[i, 1])
        elif r > 0.65:
            data.append(data2.iloc[i, 1])
        else:
            data.append(data3.iloc[i, 1])
final = pd.DataFrame()
print(data)
final = final.append(data)
final.to_csv('merge_final_8.csv')



