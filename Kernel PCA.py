#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt 
import sys

filename = sys.argv[1]
alpha = float(sys.argv[2])
spread = float(sys.argv[3])

data = pd.read_csv(filename)
data = data.iloc[:,1:-1]
data = data[:5000]
data = np.matrix(data)

n = np.shape(data)[0]
d = np.shape(data)[1]
k = data*np.transpose(data)

#kpca
m = (np.identity(n)-1/n*np.ones([n,n]))

kbar = m*k*m
value,vector = np.linalg.eig(kbar)
lbd = value/n

print(value[:2])

var = sum(lbd)
var2 = 0
i = 0
while var2<alpha*var:
    var2+=lbd[i]
    i+=1
print(i)

v1 = lbd[:2]
v2 = vector[:,:2]

print(v1)
print(v2)

v2[:,0]=v2[:,0]/np.sqrt(value[0])
v2[:,1]=v2[:,1]/np.sqrt(value[1])

a = np.transpose(v2)*kbar

fig = plt.figure()       
plt.scatter(a[0].tolist(),a[1].tolist())  
plt.show() 

#pca
mu1 = np.sum(data,axis = 0)/n
central = data-mu1
cov = np.dot(np.transpose(central),central)/n
value1,vector1 = np.linalg.eigh(cov)

print(value1[-2:])
print(np.sum(value1[-2:])/np.sum(value1))

axis1 = (data-mu1)*vector1[:,-1]
axis2 = (data-mu1)*vector1[:,-2]

fig = plt.figure()     
plt.scatter(list(axis1),list(axis2))  
plt.show() 

#gaussian kernel
def plota(spread):

    l = []
    for i in range(n):
        l.append(np.linalg.norm(data[i])**2)
    s = np.reshape(l,(1,n))
    st = np.reshape(l,(n,1))
    a = s+st-2*data*np.transpose(data)
    k1 = np.exp(-a/spread/2)
    
    m = (np.identity(n)-1/n*np.ones([n,n]))
    kbar1 = m*k1*m
    
    value2,vector2 = np.linalg.eig(kbar1)
    
    lbd1 = value2/n
    v1 = lbd1[:2]
    vector2 = vector2[:,:2]
    print(v1)
    print("{:.3f}".format(np.sum(v1)/np.sum(lbd1)))
    vector2[:,0]=vector2[:,0]/np.sqrt(value2[0])
    vector2[:,1]=vector2[:,1]/np.sqrt(value2[1])
    vector2 = np.transpose(np.mat(vector2))
    kbar1 = np.mat(kbar1)
    a1 = vector2*kbar1
    fig = plt.figure()       
    plt.scatter(a1[0].tolist(),a1[1].tolist())  
    plt.show() 
    
plota(1)

plota(100)

plota(1000)

plota(10000)

plota(100000)

plota(10000000)

plota(40000000)

plota(50000000)

plota(100000000)

plota(spread)

