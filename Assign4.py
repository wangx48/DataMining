#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import sys
import copy
import random

filename = sys.argv[1]
eta = float(sys.argv[2])
eps = float(sys.argv[3])
maxiter = int(sys.argv[4])

#data = pd.read_csv('energydata_complete.csv')
#maxiter = 100000
#eps = 0.000000001
#eta = 0.0001/20000
data = pd.read_csv(filename)
data = data.iloc[:,1:-1]

def one_hot(x):
    if x<=30:
        return 1
    elif x<=50:
        return 2
    elif x<=100:
        return 3
    else:
        return 4

data['Appliances'] = data['Appliances'].apply(lambda x:one_hot(x))
data = np.matrix(data)

n = np.shape(data)[0]
d = np.shape(data)[1]
k = 4

y = data[:,0]
x = data[:,1:]

one = np.ones((n,1))
x = np.c_[one,x]

e = np.eye(4)[np.array(y).astype('int64')-1]

sample = int(n*0.7)

trainx = x[:sample,:]
testx = x[sample:,:]
trainy = np.matrix(e[:sample,:])
testy = np.matrix(e[sample:,:])

t = 0
w = np.matrix(np.zeros((d,k)))
diff = 1
while diff>eps and t < maxiter:
#while diff>eps:
    w1 = copy.copy(w)
    #index = np.random.choice(sample, sample, replace=False)
    trainx1 = trainx
    trainy1 = trainy
    #for i in range(sample):
    i = random.randint(0,sample-1)   
    for j in range(k-1):
       
        pi = 1/(np.exp(trainx1[i]*w1[:,0]-trainx1[i]*w1[:,j])+np.exp(trainx1[i]*w1[:,1]-trainx1[i]*w1[:,j])+np.exp(trainx1[i]*w1[:,2]-trainx1[i]*w1[:,j])+np.exp(0-trainx1[i]*w1[:,j]))
        delta = (trainy1[i,j]-pi)*trainx1[i]
     
        w1[:,j] = w1[:,j]+eta*np.transpose(delta)   
    diff = np.linalg.norm(w-w1,ord = 2)
    w = copy.copy(w1)
    t+=1

yfit = testx*w
prd = []
for i in range(n-sample):
    softmax = []
    for j in range(k):
        pi = 1/(np.exp(testx[i]*w[:,0]-testx[i]*w[:,j])+np.exp(testx[i]*w[:,1]-testx[i]*w[:,j])+np.exp(testx[i]*w[:,2]-testx[i]*w[:,j])+np.exp(0-testx[i]*w[:,j]))
        softmax.append(pi)
    prd.append(softmax.index(max(softmax))+1)

accuracy = np.sum(pd.DataFrame(data[sample:,0])[0]==prd)/(n-sample)

print(accuracy)

print(w)

