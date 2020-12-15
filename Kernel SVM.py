#!/usr/bin/env python
# coding: utf-8


#svm
#kernel:"linear", "gaussian" or "polynomial"
#loss:"hinge" or "quadratic"
#kernel_param: the spread σ2 for gaussian or a comma separated pair q,c for the polynomial kernel

import numpy as np
import pandas as pd
import sys
import copy
import random

filename = sys.argv[1]
loss = sys.argv[2]
c = float(sys.argv[3])
eps = float(sys.argv[4])
maxiter = int(sys.argv[5])
kernel = sys.argv[6]
kernel_param = sys.argv[7]

def one_hot(x):
    if x<=50:
        return -1 
    else:
        return 1
    
data = pd.read_csv(filename)
data = data.iloc[:,1:-1]

data['Appliances'] = data['Appliances'].apply(lambda x:one_hot(x))
data.iloc[:,1:] = data.iloc[:,1:].apply(lambda x:(x - np.max(x)) / ((np.max(x)- np.min(x))))
data = np.matrix(data)

n = np.shape(data)[0]
d = np.shape(data)[1]-1
sample = int(0.7*5000)

trainindex = np.random.choice(n,sample,replace = False)
testindex = [i for i in range(n) if i not in trainindex]


xtrain = data[trainindex,1:]
ytrain = data[trainindex,0]
xtest = data[testindex,1:][:5000-sample,]
ytest = data[testindex,0][:5000-sample,]



def ker(data,kernel,loss,kernel_param,c):
    
    if kernel == 'linear':
        k = data*np.transpose(data)
    elif kernel == 'gaussian':
        #spread σ2
        spread = float(kernel_param)
        l = []
        for i in range(sample):
            l.append(np.linalg.norm(data[i])**2)
        s = np.reshape(l,(1,sample))
        st = np.reshape(l,(sample,1))
        a = s+st-2*data*np.transpose(data)
        k = np.exp(-a/spread/2)
    else:
    #polynomial q,c
        q,c = map(float,kernel_param.split(',')) 
        k = (data*np.transpose(data)+c)**int(q)

    
    if loss == 'quadratic':
        for i in range(sample):
            k[i,i]+= 1/2/c
    return k



def svm(xtrain,ytrain,kernel,loss,kernel_param,c,eps,maxiter):
    
    sample = np.shape(xtrain)[0]
    k = ker(xtrain,kernel,loss,kernel_param,c)
    k = k+1
    
    step = []
    for i in range(sample):
        
        step.append(1/k[i,i])
    
    t = 0
    alpha = np.zeros((1,sample))
    gap = 1

    while t<maxiter and gap > eps:
        rand = random.randint(0,sample-1) 
        alpha1 = copy.copy(alpha)
        for j in range(sample):
           
            total = np.sum(np.multiply(np.multiply(alpha1,np.transpose(ytrain)),k[j]))

            alpha1[0][j]+=step[j]*(1-ytrain[j]*total)
            if alpha1[0][j]<0: 
                alpha1[0][j] = 0
            elif loss == 'hinge' and alpha1[0][j]>c: 
                alpha1[0][j] = c
            else:
                pass
        gap = np.linalg.norm(alpha-alpha1,ord = 2)
        alpha = copy.copy(alpha1)
        t+=1
    print('iteration:', t)
    print('gap:',gap)
    if kernel=='linear':
        if loss == 'hinge':
            w = np.zeros((1,d))
            b = 0
            cnt = 0
            for i in range(sample):
                if alpha[0][i]>0:
                    w+=alpha[0][i]*ytrain[i][0]*xtrain[i]
                    cnt+=1
            for j in range(sample):
                if alpha[0][j]<c and alpha[0][j]>0:
                    b+=ytrain[j][0]-w*np.transpose(xtrain[j])
            b = b/cnt
        else:
            w = np.zeros((1,d))
            b = 0
            cnt = 0
            for i in range(sample):
                if alpha[0][i]>0:
                    w+=alpha[0][i]*ytrain[i][0]*xtrain[i]
                    cnt+=1
            for j in range(sample):
                if alpha[0][j]>0:
                    b+=ytrain[j][0]-w*np.transpose(xtrain[j])
            b = b/cnt
        print('w: ',w)
        print('b: ',b)
    return alpha


def test(a,xtest,ytest,xtrain,ytrain,kernel,loss,kernel_param,c):
    
    y = []
    testsize = np.shape(xtest)[0]
    trainsize = np.shape(xtrain)[0]
    
    number = 0
    
    if loss == 'hinge':
        
        for i in range(trainsize):
            if a[0][i]>0 and a[0][i]<c:
              
                number+=1
    else:
        for i in range(trainsize):
            if a[0][i]>0:
          
                number+=1
    print('There are',number,'support vectors')   
    
    if kernel =='linear':
        for i in range(testsize):
            total = 0
            for j in range(trainsize):
                if a[0][j]>0:
                    total+=a[0][j]*ytrain[j][0]*(xtest[i]*np.transpose(xtrain[j])+1)
     
            #total = np.sum(np.multiply(np.multiply(a,np.transpose(ytrain)),xtest[i]*np.transpose(xtrain)+1))
            y.append(int(np.sign(total)))
           
                
    elif kernel == 'gaussian':
        spread = float(kernel_param)
        for i in range(testsize):
            total = 0
            for j in range(trainsize):
                if a[0][j]>0:
                    total+=a[0][j]*ytrain[j][0]*(np.exp(np.square(np.linalg.norm(xtest[i]-xtrain[j]))*(-1)/2/spread)+1)
            
            y.append(int(np.sign(total)))
    else:
        q,c = map(float,kernel_param.split(','))
        for i in range(testsize):
            total = 0
            for j in range(trainsize):
                if a[0][j]>0:
                    total+=a[0][j]*ytrain[j][0]*((c+xtest[i]*np.transpose(xtrain[j]))**int(q)+1)
                
            y.append(int(np.sign(total)))      
    accuracy = np.sum(pd.DataFrame(ytest)[0]==y)/testsize
    print('accuracy:',accuracy)





#a = svm(xtrain,ytrain,'gaussian','hinge',0.1,1,0.00001,2000)

#test(a,xtest,ytest,xtrain,ytrain,'gaussian','hinge',0.1,1)




a=svm(xtrain,ytrain,kernel,loss,kernel_param,c,eps,maxiter)
test(a,xtest,ytest,xtrain,ytrain,kernel,loss,kernel_param,c)

