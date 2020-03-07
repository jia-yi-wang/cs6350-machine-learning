#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


train_data = np.loadtxt('/Users/wjy/Downloads/bank-note/train.csv',delimiter=',')


# In[7]:


test_data = np.loadtxt('/Users/wjy/Downloads/bank-note/test.csv',delimiter=',')


# In[42]:


def pred_error(weight,X,y):
    num = X.shape[0]
    y_h = np.squeeze(np.dot(X,weight.T))
    d = np.multiply(y_h,y)
    d[d<0] = 0
    d[d>0] = 1
    d = 1 - d
    err = np.sum(d)
    r = err/num
    return r


# In[71]:


def pred_error_vote(C,W,X,y):
    num = X.shape[0]
    count = 0
    Cn = np.array(C)
    Wn = np.array(W)
    Wn = np.squeeze(Wn)
    for i in range(0,num):
        y_h = np.dot(X[i,:],Wn.T)
        y_h[y_h>0] = 1
        y_h[y_h<=0] = -1
        f = np.dot(np.squeeze(y_h),Cn)
        if f * y[i] < 0:
            count = count + 1
    err = count / num
    return err


# In[19]:


dim = train_data.shape[1]
train_x = train_data[:,0:dim-1]
train_x = np.insert(train_x,0,values=1,axis=1)
train_y = train_data[:,dim-1]
train_y = 2*train_y - 1
test_x = test_data[:,0:dim-1]
test_x = np.insert(test_x,0,values=1,axis=1)
test_y = test_data[:,dim-1]
test_y = 2*test_y - 1


# In[ ]:


### standard perceptron
num = train_x.shape[0]
w = np.zeros([1,dim])
r = 0.5
t = 0
T = 10
for k in range(0,T):
    for i in range(0,num):
        y_e = np.dot(w,train_x[i,:])
        if y_e * train_y[i] <= 0:
            w = w + r * train_y[i] * train_x[i,:]
    error = pred_error(w,test_x,test_y)
    print('T=',k+1,'Error=',error,'weight vector:',w)


# In[ ]:


### voted perceptron
num = train_x.shape[0]
w = np.zeros([1,dim])
r = 0.5
t = 0
T = 10
W = []
C = []
Cm = 0
for k in range(0,T):
    for i in range(0,num):
        y_e = np.dot(w,train_x[i,:])
        if y_e * train_y[i] <= 0:
            W.append(w)
            C.append(Cm)
            w = w + r * train_y[i] * train_x[i,:]
            Cm = 1
        else:
            Cm = Cm + 1
    error = pred_error_vote(C,W,test_x,test_y)
    print('T=',k+1,'test error:',error)
print('counts:',C)
print('weight vector:',np.array(W))


# In[ ]:


### averaged perceptron
num = train_x.shape[0]
w = np.zeros([1,dim])
a = np.zeros([1,dim])
r = 0.5
t = 0
T = 10
for k in range(0,T):
    for i in range(0,num):
        y_e = np.dot(w,train_x[i,:])
        if y_e * train_y[i] <= 0:
            w = w + r * train_y[i] * train_x[i,:]
    a = a + w
    error = pred_error(a,test_x,test_y)
    print('T=',k+1,'Error=',error,'weight vector:',a)


# In[74]:





# In[ ]:




