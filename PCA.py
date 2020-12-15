#!/usr/bin/env python
# coding: utf-8


#part1
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

filename = sys.argv[1]
alpha = float(sys.argv[2])

data = pd.read_csv(filename)
data = data.iloc[:,1:-1]
data = np.matrix(data)
n = np.shape(data)[0]

mu = np.sum(data,axis = 0)/n
mu = np.array(mu.tolist()[0])
central = data-mu
cov = np.dot(np.transpose(central),central)/n
value,vector = np.linalg.eigh(cov)

total= np.sum(value)
total2 = 0
i = 0
while total2<alpha*total:
    total2+=value[26-i]
    i+=1

#dimensions required to capture α
print(i)

#mean squared error
mse3 = value[26]+value[25]+value[24]
print(mse3)


axis1 = (data-mu)*np.transpose(vector[-1])
axis2 = (data-mu)*np.transpose(vector[-2])
axis3 = (data-mu)*np.transpose(vector[-3])

fig = plt.figure()     
ax1 = plt.axes(projection='3d')
ax1.scatter3D(list(axis1),list(axis2),list(axis3))  
plt.show() 

