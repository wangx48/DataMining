#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt 
import sys

filename = sys.argv[1]
data = pd.read_csv(filename)
#data = pd.read_csv('energydata_complete.csv')
data = data.iloc[:,1:-1]
data = np.matrix(data)

n = np.shape(data)[0]
d = np.shape(data)[1]

sample = int(n*0.7)
train = data[:sample,:]
test = data[sample:,:]
y = train[:,0]
x = train[:,1:]
testy = test[:,0]
testx = test[:,1:]

one = np.ones((sample,1))
x = np.c_[one,x]
one = np.ones((n-sample,1))
testx = np.c_[one,testx]

q = np.zeros_like(x)
r = np.zeros((d,d))
for i in range(d):
    da = x[:,i]
   
    for j in range(i):
        r[j][i]=np.transpose(da)*q[:,j]/(np.transpose(q[:,j])*q[:,j])
        da = da-r[j][i]*q[:,j]
    
    r[i,i]=1
    q[:,i]=da

delta = np.zeros((d,d))
for i in range(d):
    delta[i][i] = 1/np.sum(np.square(q[:,i]))
    
right = delta*np.transpose(q)*y

w = np.zeros((d,1))
for i in range(d-1,-1,-1):
    w[i] = right[i]-np.dot(r[i,:],w)

print('w',w)
yfit = x*w
sse = np.sum(np.square(y-yfit))
tss = np.sum(np.square(y-np.mean(y)))
r2 = (tss-sse)/tss
print('train')
print('sse ',sse)
print('mse',sse/sample)
print('tss',tss)
print('r2',r2)

ypre = testx*w
sse = np.sum(np.square(testy-ypre))
tss = np.sum(np.square(testy-np.mean(testy)))
r2 = (tss-sse)/tss
print('test')
print('sse ',sse)
print('mse',sse/(n-sample))
print('tss',tss)
print('r2',r2)

l2 = np.linalg.norm(w)
print('l2',l2)

