# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:49:51 2021

@author: SANDEEP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
print(x)
print(y)





