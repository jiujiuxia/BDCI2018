# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:45:20 2018

@author: jiuxia
"""

import numpy as np

def read_data(file_name):
    """This function is taken from:
    https://github.com/benhamner/BioResponse/blob/master/Benchmarks/csv_io.py
    """
    f = open(file_name)
    #ignore header
    f.readline()
    samples = []
    target = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line]
        samples.append(sample)
    return samples

def load():
    """Conveninence function to load all data as numpy arrays.
    """
    print("Loading data...")
    filename_train = 'train1012_4.0.csv'
    filename_test = 'test1012_4.0.csv'

    train = read_data(filename_train)
    y_train = np.array([x[-1] for x in train])
    X_train = np.array([x[1:-1] for x in train])
    X_test = np.array(read_data(filename_test))
    X_test = np.array([x[1:] for x in X_test])
    return X_train, y_train, X_test

if __name__ == '__main__':

    X_train, y_train, X_test = load()
    print(X_train)
    print(y_train)
    print(X_test)