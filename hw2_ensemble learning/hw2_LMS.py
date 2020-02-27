#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def pick_batch(X,Y,batch_size):
    num = X.shape[0]
    ind = np.random.randint(num,size=batch_size)
    X_p = X[ind,:]
    y_p = Y[ind]
    return X_p,y_p


# In[3]:


def compute_gradient(X,y,w):
    num = X.shape[0]
    dim = X.shape[1]
    dw = np.zeros([1,dim])
    for i in range(0,num):
        dw = dw + (np.dot(X[i,:],w.T)-y[i])*X[i,:]
    return dw


# In[4]:


def load_data(): #X.shape = [num,dim]
    data = np.loadtxt('train.csv',delimiter=',')
    dim = data.shape[1]
    num = data.shape[0]
    X = data[:,0:dim-1]
    X = np.insert(X,0,values=np.ones(num),axis=1)
    y = data[:,dim-1]
    return X,y


# In[5]:


def load_test(): #X.shape = [num,dim]
    data = np.loadtxt('test.csv',delimiter=',')
    dim = data.shape[1]
    num = data.shape[0]
    X = data[:,0:dim-1]
    X = np.insert(X,0,values=np.ones(num),axis=1)
    y = data[:,dim-1]
    return X,y


# In[6]:


def compute_LMS_loss(w,X,y):#w.shape=[1,dim]#
    loss = 0.5 * np.sum(np.square(np.dot(w,X.T)-y))
    return loss


# In[7]:


def batchSGD_LMS(X,y,w0,max_iters,learning_rate,batch_size):
    w = w0
    loss = np.zeros(max_iters)
    for i in range(0,max_iters):
        flag = 0
        #X_p,y_p = pick_batch(X,y,batch_size)
        dw = compute_gradient(X,y,w)
        w = w - learning_rate * dw
        loss[i] = compute_LMS_loss(w,X,y)
        print(loss[i])
    return w,loss


# In[176]:


data_X,data_y = load_data()
w0 = np.zeros([1,data_X.shape[1]])
max_iters = 5000
learning_rate  = 0.01
batch_size = 1
w,loss = batchSGD_LMS(data_X,data_y,w0,max_iters,learning_rate,batch_size)
t_X,t_y = load_test()
t_loss = compute_LMS_loss(w,t_X,t_y)
print(t_loss)


# In[171]:


plt.plot(loss)


# In[177]:


print(w)


# In[141]:


#compute result analyticallt

num = X.shape[0]
dim = X.shape[1]
Y = np.zeros([num,1])
Y[:,0] = y
ww = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
print(ww.T)


# In[ ]:




