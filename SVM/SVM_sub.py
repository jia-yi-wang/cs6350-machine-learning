#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


# In[3]:


def svm(x,y,w,gamma,lam):
    if (y*np.dot(w,x)) < 1:
        w = (1-gamma) * w + gamma * lam * y * x
    else:
        w = (1-gamma) * w      
    return w


# In[83]:


def svm_dual(alpha,X,Y):
    t1 = np.multiply(alpha,Y)
    Q = np.dot(X,X.T)
    f = np.dot(t1,np.dot(Q,t1.T))*0.5 - np.sum(alpha)
    return f


# In[161]:


def svm_kernel_dual(alpha,X,Y):
    t1 = np.multiply(alpha,Y)
    Q = K
    f = np.dot(t1,np.dot(Q,t1.T))*0.5 - np.sum(alpha)
    return f


# In[159]:


def gauss_kernel(X):
    num = X.shape[0]
    K = np.zeros([num,num])
    for i in range(0,num):
        for j in range(0,i+1):
            K[i,j] = np.exp(-np.linalg.norm(X[i,:]-X[j,:])**2/const)
            K[j,i] = K[i,j]
    return K


# In[186]:


def predict_kernel(X1,X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros([n1,n2])
    for i in range(0,n1):
        for j in range(0,n2):
            K[i,j] = np.exp(-np.linalg.norm(X1[i,:]-X2[j,:])**2/const)
    return K


# In[117]:


def c1(alpha):
    f = np.dot(alpha,Y)
    return f


# In[86]:


def c2(alpha):
    f = C-alpha
    return f


# In[29]:


train_data = np.loadtxt('/Users/wjy/Downloads/bank-note/train.txt')
test_data = np.loadtxt('/Users/wjy/Downloads/bank-note/test.txt')
train_data = np.insert(train_data, 0, values=1, axis=1)
test_data = np.insert(test_data, 0, values=1, axis=1)


# In[10]:


def cmp_loss(w,Data,lam):
    N = Data.shape[0]
    d = Data.shape[1]
    loss = 0
    for i in range(0,N):
        judge = 1-Data[i,d-1]*np.dot(w,Data[i,0:d-1])
        if judge > 0:
            loss = loss + lam * judge
    loss = loss + 0.5 * np.dot(w,w)
    return loss


# In[36]:


def cmp_error(w,Data):
    N = Data.shape[0]
    d = Data.shape[1]
    error = 0
    for i in range(0,N):
        if Data[i,d-1]*np.dot(w,Data[i,0:d-1]) < 0:
            error = error + 1
    rate = error/N
    return rate


# In[205]:


def cmp_kernel_error(alpha,train,test):
    N = test.shape[0]
    dd = test.shape[1]
    Ker = predict_kernel(train[:,0:dd-1],test[:,0:dd-1])
    Y = np.dot(np.multiply(alpha,train[:,dd-1]),Ker)
    error = 0
    for i in range(0,N):
        if test[i,dd-1]*Y[i] < 0:
            error = error + 1
    rate = error/N
    return rate


# In[212]:


C = 100/873
gamma_0 = 2
t = 1
epoch = 5
dim = train_data.shape[1]
num = train_data.shape[0]
w = np.zeros(dim-1)
Loss = []
const = 0.1
#for i in range(0,epoch):
#    np.random.shuffle(train_data)
#    for j in range(0,num):
#        gamma = gamma_0 / (t + 1)
#        w = svm(train_data[j,0:dim-1],train_data[j,dim-1],w,gamma,C)
#        t = t + 1
#        loss = cmp_loss(w,train_data,C)
#        Loss.append(loss)


# In[65]:


Loss = np.array(Loss)
plt.plot(Loss)
#print(train_data[0:10,:])


# In[67]:


er = cmp_error(w,test_data)
print(er)


# In[213]:


####construct svm dual optimization problem
X = train_data[:,0:dim-1]
Y = train_data[:,dim-1]
K = gauss_kernel(X)
cons = {'type':'eq','fun':c1}
buds = [[0,C]]*num
alpha0 = np.zeros(num)
result=opt.minimize(fun=svm_kernel_dual,x0=alpha0,args=(X,Y),method='L-BFGS-B',bounds=buds,constraints=cons)
alpha = result.x


# In[214]:


#w = np.dot(X.T,np.multiply(alpha,Y))
train_error = cmp_kernel_error(alpha,train_data,train_data)
test_error = cmp_kernel_error(alpha,train_data,test_data)
#print(w)
print(train_error)
print(test_error)


# In[215]:


print(alpha)


# In[ ]:




