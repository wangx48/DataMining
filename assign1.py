#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys


# In[2]:


filename = sys.argv[1]
eps = float(sys.argv[2])


# In[2]:


data = pd.read_csv(filename)
#data = pd.read_csv('energydata_complete.csv')
data = data.iloc[:,1:-1]
data = np.matrix(data)
n = np.shape(data)[0]


# In[3]:


#a
mu = np.sum(data,axis = 0)/n
mu = np.array(mu.tolist()[0])
print(mu)


# In[4]:


var = np.sum(np.square(data-mu))/n
print(var)


# In[5]:


#b
central = data-mu
cov1 = np.dot(np.transpose(central),central)/n
print(cov1)


# In[6]:


cov2 = np.zeros((27,27))
for i in range(n):
    cov2 = cov2+np.transpose(data[i]-mu)*(data[i]-mu)
cov2 = cov2/n
print(cov2)


# In[7]:


#c
corr = np.zeros((27,27))
l = []
for k in range(27):
    l.append(np.sqrt(np.sum(np.square(data[:,k]-mu[k]))))
    k+=1 
for i in range(27):
    vec1 = central[:,i]
    for j in range(27):
        vec2 = central[:,j]
        corr[i][j] = np.transpose(vec1)*vec2/l[i]/l[j]
        j+=1
    i+=1
print(corr)


# In[37]:


ma = 0
mi = 0
ab = 1
for m in range(1,27):
    for n in range(m):
        number = corr[m][n]
        if number>ma:
            ma = number
        if number<mi:
            mi = number
        if abs(number)<abs(ab):
            ab = number
for m in range(1,27):
    for n in range(m):
        if corr[m][n]==ma:
            print('most correlated',m,n,ma)
        if corr[m][n]==mi:
            print('most anti-correlated',m,n,mi)
        if abs(corr[m][n])==abs(ab):
            print('least correlated',m,n,ab)
            
# 21th and 13th attributes are the most correlated
# 14th and 15th attributes are the most anti-correlated
# 1st and 25th attributes are the least correlated


# In[9]:


#the most correlated
n = np.shape(data)[0]
fig = plt.figure()       
plt.scatter(data[:,12].tolist(),data[:,20].tolist())  
plt.show() 


# In[10]:


#the most anti-correlated
n = np.shape(data)[0]
fig = plt.figure()      
plt.scatter(data[:,13].tolist(),data[:,14].tolist())  
plt.show() 



# In[38]:


#the least correlated
n = np.shape(data)[0]
fig = plt.figure()      
plt.scatter(data[:,0].tolist(),data[:,24].tolist())  
plt.show() 



# In[12]:


#CSCI-6390 Only: First Two Eigenvectors and Eigenvalues
def fun(x):
    x = cov1*x
    a = x[:,0]
    b = x[:,1]
    b = b - float(np.transpose(b)*a/(np.transpose(a)*a))*a
    a = a/np.sqrt(np.sum(np.square(a)))
    b = b/np.sqrt(np.sum(np.square(b)))
    return a*np.matrix([1,0])+b*np.matrix([0,1])

def distance(a,b):
    return np.sqrt(np.sum(np.square(a-b)))


# In[15]:


x = cov1[:,0:2]
while distance(x,fun(x))>eps:
    x = fun(x)
print(x)


# In[18]:


u1 = x[:,0]
u2 = x[:,1]
print('eigen vectors')
print(u1)
print(u2)

v1 = float((cov1*u1)[0]/u1[0])
v2 = float((cov1*u2)[0]/u1[0])
print('eigen values')
print(v1)
print(v2)

axis1 = (data-mu)*u1
axis2 = (data-mu)*u2

fig = plt.figure()      
plt.scatter(list(axis1),list(axis2))  
plt.show() 





