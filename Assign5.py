#!/usr/bin/env python
# coding: utf-8

# In[179]:


import numpy as np
import pandas as pd
import sys
import random

filename = sys.argv[1]
eta = float(sys.argv[2])
maxiter = int(sys.argv[3])
hiddensize = int(sys.argv[4])
numhidden = int(sys.argv[5])


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
sample = int(0.7*n)
d = np.shape(data)[1]-1
k = 4

trainindex = np.random.choice(n,sample,replace = False)
testindex = [i for i in range(n) if i not in trainindex]


e = np.eye(4)[np.array(y).astype('int64')-1]

xtrain = data[trainindex,1:]
xtest = data[testindex,1:] 
ytrain = e[trainindex,0]
ytest = e[testindex,0]


# In[180]:


def relu(x):
    i = np.shape(x)[0]
    j = np.shape(x)[1]
    for k in range(i):
        for t in range(j):
            if x[k][t] >=0:
                pass
            else:
                x[k][t]=0
    return x


# In[181]:


def softmax(x):
    o = np.exp(x)
    total = np.sum(o)
    return o/total


# In[182]:


def relug(x):
    i = np.shape(x)[0]
    j = np.shape(x)[1]
    for k in range(i):
        for t in range(j):
            if x[k][t] >0:
                x[k][t]=1
            else:
                x[k][t]=0
    return x


# In[183]:


def mlp(maxiter,hiddensize,numhidden,eta):
    b = []
    w = []
    #b.append(np.ones((hiddensize,1)))
    #w.append(np.ones((d,hiddensize)))
    b.append(np.random.rand(hiddensize,1))
    w.append(np.random.rand(d,hiddensize))
    for l in range(numhidden-1):
        b.append(np.random.rand(hiddensize,1))
        w.append(np.random.rand(hiddensize,hiddensize))
    b.append(np.random.rand(k,1))
    w.append(np.random.rand(hiddensize,k))
    
    #    b.append(np.ones((hiddensize,1)))
    #    w.append(np.ones((hiddensize,hiddensize)))
    #b.append(np.ones((k,1)))
    #w.append(np.ones((hiddensize,k)))
    
    t = 0
    while t <maxiter:
#feed-forward
        i = random.randint(0,sample-1)
        z = []
        z.append(np.transpose(xtrain[i]))
        for l in range(numhidden+1):
            z.append(relu(b[l]+np.transpose(w[l])*z[l]))
#backpropagation
        delta = []
        o = z[-1]
        o1 = np.ones((4,1))
        for g in range(4):
            o1[g] = 1/(np.exp(o[0]-o[g])+np.exp(o[1]-o[g])+np.exp(o[2]-o[g])+np.exp(o[3]-o[g]))
        delta.append(o1-np.transpose(np.matrix(ytrain[i])))
        
        for i in range(numhidden):

            delta.append(np.multiply(w[-(i+1)]*delta[i],relug(z[-i-2])))
            
        for j in range(numhidden):
            w[j] = w[j]-eta*z[j]*np.transpose(delta[-j-1])
            b[j] = b[j]-eta*delta[-j-1]
        t+=1

#test
    tran = xtrain
    test = np.shape(ytest)[0]
    prd = []
    for a in range(test):
        tran = np.transpose(xtest[a])
        for i in range(numhidden+1):
            tran = np.transpose(w[i])*tran+b[i]
            tran = relu(tran)
        yfit = softmax(tran).tolist()
        prd.append(yfit.index(max(yfit))+1)
    accuracy = np.sum(pd.DataFrame(data[testindex,0])[0]==prd)/(n-sample)

    return(accuracy)




# In[184]:


mlp(100,5,5,0.01)


# In[185]:


mlp(100,5,10,0.01)


# In[186]:


mlp(100,10,5,0.01)


# In[187]:


mlp(100,10,10,0.01)


# In[189]:


mlp(100,5,5,0.001)

mlp(maxiter,hiddensize,numhidden,eta)
  