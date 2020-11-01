#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import sys
import random

filename = sys.argv[1]
eta = float(sys.argv[2])
maxiter = int(sys.argv[3])
hiddensize = int(sys.argv[4])
numhidden = int(sys.argv[5])

#filename = 'energydata_complete.csv'
data = pd.read_csv(filename)
data = data.iloc[:,1:-1]
data2 = (data - data.min()) / (data.max() - data.min())
def one_hot(x):
    if x<=30:
        return 0
    elif x<=50:
        return 1
    elif x<=100:
        return 2
    else:
        return 3

def one_hot2(x):
    if x<100:
        return int(x/10)
    else:
        return 10
    
data['Appliances'] = data['Appliances'].apply(lambda x:one_hot2(x))
data = np.matrix(data)
data2 = np.matrix(data2)

n = np.shape(data)[0]
sample = int(0.7*n)
d = np.shape(data)[1]-1
k = 11

y=data[:,0]
trainindex = np.random.choice(n,sample,replace = False)
testindex = [i for i in range(n) if i not in trainindex]


e = np.eye(k)[np.array(y).astype('int64')]

xtrain = data[trainindex,1:]
xtest = data[testindex,1:] 
ytrain = e[trainindex,0]
ytest = e[testindex,0]




def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    o = np.exp(x)
    total = np.sum(o)
    return o/total

def get_loss(scores, labels):
    scores_norm = scores- np.max(scores, axis=1)
    scores_norm = np.exp(scores_norm)
    scores_norm = scores_norm / np.sum(scores_norm, axis=1) 
  
    true = labels.reshape(-1).argmax()

    true_class_scores = scores_norm.max()
    loss = np.mean(-np.log(true_class_scores))
    one_hot = np.zeros(scores.shape)
    one_hot[0][true] = 1.0
    grad = scores_norm - one_hot

    return loss, grad



def relug(x):
    return np.maximum(x, 0)



def mlp(maxiter,hiddensize,numhidden,eta):
    b = []
    w = []
    #b.append(np.ones((hiddensize,1)))
    #w.append(np.ones((d,hiddensize)))
    b.append(np.ones((1, hiddensize)))
    w.append(np.ones((d,hiddensize)))
    for l in range(numhidden-1):
        b.append(np.ones((1, hiddensize)))
        w.append(np.ones((hiddensize,hiddensize)))
    b.append(np.ones((1, k)))
    w.append(np.ones((hiddensize,k)))
    
    #    b.append(np.ones((hiddensize,1)))
    #    w.append(np.ones((hiddensize,hiddensize)))
    #b.append(np.ones((k,1)))
    #w.append(np.ones((hiddensize,k)))
    
    t = 0 
    ddb = [0 for i in range(numhidden+1)]
    ddw = [0 for i in range(numhidden+1)]
    update = 0

    while t <maxiter:
#feed-forward
        i = random.randint(0,sample-1)
        z = []
        #print(xtrain[i].shape)
        z.append(xtrain[i]/10)
        #z.append(xtrain[i])
        for l in range(numhidden):
            #print(np.matmul(z[l], w[l]).shape, b[l].shape)
            out = np.matmul(z[l], w[l]) + b[l]
            z.append(relu(out))
        output = np.matmul(z[-1], w[-1]) + b[-1]

#bp
        loss, grad = get_loss(output, ytrain[i])
        #print(grad, w[-1].T)
        db = np.sum(grad, axis=0)
        dW = np.matmul(z[-1].T, grad)
        dX = np.matmul(grad, w[-1].T)
        b[-1] -= db*eta
        w[-1] -= dW*eta
        for i in reversed(range(numhidden)):
            #dX = np.multiply(dX,np.multiply (z[i+1] , 1-z[i+1]) )
            dX = np.multiply(dX,(z[i + 1] > 0))
            db = np.sum(dX, axis=0)
            dW = np.matmul(z[i].T, dX)
            dX = np.matmul(dX, w[i].T)
            b[i] -= db*eta
            w[i] -= dW*eta
        update +=1
        #print(len(ddb))
        if(update == 100):
            update = 0
            #eta *= 0.9
            #print(t, loss)
            #print(z[0])

        t += 1
        


#test  np.argmax(scores, axis=1)
    test = np.shape(ytest)[0]
    prd = []
    correct = 0
    for a in range(test):
        #tran = xtest[a]
        tran = xtest[a] /10
        for i in range(numhidden):
            tran = np.matmul(tran , w[i]) + b[i]
            tran = relu(tran)
        out =   np.matmul(tran , w[-1] ) + b[-1]
        out =   out.reshape(-1)
        prd = np.argmax(out)
        label = np.argmax(ytest[a])
        if(prd <=2):
            prd = 0
        elif(prd <=4):
            prd = 1
        elif(prd <=9):
            prd = 2
        else:
            prd = 3
        if(label <=2):
            label = 0
        elif(label <=4):
            label = 1
        elif(label <=9):
            label = 2
        else:
            label = 3
        correct += (prd == label)
    accuracy = correct / float(test)
    print('bias')
    print(b)
    print('weight')
    print(w)
    return(accuracy)




#print(mlp(30000,100,2,1))

print(mlp(maxiter,hiddensize,numhidden,eta))

