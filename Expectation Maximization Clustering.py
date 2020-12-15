#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from numpy.linalg import det
from numpy.linalg import inv
import copy
import sys

filename = sys.argv[1]
k = int(sys.argv[2])
eps = float(sys.argv[3])
ridge = float(sys.argv[4])
maxiter = int(sys.argv[5])

#filename = 'energydata_complete.csv'
data = pd.read_csv(filename)
data = data.iloc[:,1:-1]

def one_hot(x):
    if x<=40:
        return 0
    elif x<=50:
        return 1
    elif x<=60:
        return 2
    elif x<=90:
        return 3
    elif x<=160:
        return 4
    else:
        return 5

data['class'] = data['Appliances'].apply(lambda x:one_hot(x))

data.iloc[:,1:-1] = data.iloc[:,1:-1].apply(lambda x:(x - np.max(x)) / ((np.max(x)- np.min(x))))

train = np.matrix(data.iloc[:,1:-1])
n = len(train)
d = np.shape(train)[1]

t = 0
#k = 6
#eps = 0.001
#ridge = 0.00000001
#maxiter = 20
initial = np.random.choice(n, k, replace=False)


mu=train[initial]
mu1 = np.matrix(np.zeros((k,d)))
w = np.zeros((k,n))
p = np.ones((6,1))/k
var = [np.identity(d) for i in range(k)]
diff = 1

    
while diff>eps and t<maxiter:
    for j in range(n):
        log = []
        for i in range(k):
            sub = train[j]-mu[i]
            a = float(-np.log(np.sqrt(det(var[i]+ridge*np.identity(d))))+np.log(p[i])-sub*inv(var[i]+ridge*np.identity(d))*np.transpose(sub)/2)
            log.append(a)
      
        m = np.max(log)
       
        under = np.exp(log[0]-m)+np.exp(log[1]-m)+np.exp(log[2]-m)+np.exp(log[3]-m)+np.exp(log[4]-m)+np.exp(log[5]-m)
     
        for i in range(k):
            
            w[i,j] = np.exp(log[i]-m)/under
                              
    for i in range(k):
    
        mu1[i] = w[i]*train/np.sum(w[i])
        dbar = train-mu1[i]
        total = 0
        for j in range(n):
            total+=w[i,j]*np.transpose(dbar[j])*dbar[j]
        var[i] = total/np.sum(w[i])
        p[i] = sum(w[i])/n

    diff = np.linalg.norm(mu1-mu)

    print(diff)
    mu = mu1.copy()
    t+=1

print('final mean: \n',mu)


print('final covariance matrix:\n',var)

cls= np.argmax(w, axis=0)
result = pd.DataFrame({'cls':cls})
print('Size of each cluster: \n',result['cls'].value_counts().sort_values())

#data['class'].value_counts().sort_values()

purity = 0
for i in range(k): # label
    num = 0
    prd = result[result['cls'] == i].index
    for j in range(6): # true number

        true = data[data['class'] == j].index

        union = len(list(set(prd)&set(true)))
        num = max(num,union)
    purity+=num 
purity = purity/n 
print('purity score: ',purity/n)

