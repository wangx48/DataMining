#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import copy
import random
import sys 

filename = sys.argv[1]
k = int(sys.argv[2])
eps = float(sys.argv[3])
spread = float(sys.argv[4])

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



def union(label,label1):
    total = 0
    for i in range(k):
        l = [a for a, x in enumerate(label) if x == i]
        l1 = [a for a, x in enumerate(list(label1)) if x == i]
        u = len(list(set(l)&set(l1)))
        total+=u
    return 1-total/n

def gaussian(data,spread):
    l = []

    for i in range(n):
        l.append(np.linalg.norm(train[i])**2)
    s = np.reshape(l,(1,n))
    st = np.reshape(l,(n,1))
    a = s+st-2*train*np.transpose(train)
    return np.exp(-a/spread/2)


diff = 1
#eps = 0.000001
#spread = 10
#k = 6
t = 0

#kernel = train*np.transpose(train)
kernel = gaussian(train,spread)
label = [random.randint(0,5) for i in range(n)]
while diff>eps:

    sqnorm = []
    avg = np.zeros((n,k))

    for i in range(k):
        index = [a for a, x in enumerate(label) if x == i]
        n1 = len(index)

        sqnorm.append(np.sum(kernel[index][:,index])/n1**2)

        for j in range(n):
            avg[j][i] = np.sum(kernel[j,index])/n1
    distance = np.matrix(sqnorm)-2*avg
    label1 = np.argmin(distance, axis=1)
    diff = union(label,label1)
    label = label1.copy()
    print(diff)
    t+=1

label = [int(i) for i in label]
result = pd.DataFrame({'cls':label})
print('Size of each cluster: \n',result['cls'].value_counts().sort_values())


purity = 0
for i in range(k): # label
    num = 0
    for j in range(6): # true number
        prd = result[result['cls'] == i].index
        true = data[data['class'] == j].index

        u = len(list(set(prd)&set(true)))
        num = max(num,u)
    purity+=num 
    
print('purity score: ',purity/n)
   


# In[2]:


#10 0.30
#100 0.2747757790727134
#1000000+ 0.25

