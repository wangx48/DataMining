#!/usr/bin/env python
# coding: utf-8

# In[4]:


#part2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import random

#part2
n=100000

def dmf(d):
    l=[]
    for k in range(n):
        v1 = []
        v2 = []
        for j in range(d):
            v1.append(random.choice((-1,1)))
            v2.append(random.choice((-1,1)))
        v1 = np.matrix(v1)
        v2 = np.matrix(v2)
        cos = float(v1*np.transpose(v2))/d
        l.append(np.arccos(cos)/np.pi*180)
    print('min',min(l))
    print('max',max(l))
    print('value range',min(l),'~',max(l))
    print('mean',np.mean(l))
    print('variance',np.var(l))
    
    df = pd.DataFrame(l)
    df = pd.value_counts(df[0])
    df = pd.DataFrame(df)
    df = df.sort_index()
    val = list(df.index)
    p = list(df.values)
    p = [x/n for x in p]
    plt.plot(val,p)
    plt.show()
    
dmf(10)
dmf(100)
dmf(1000)

