#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import copy
import pdb
import matplotlib.pyplot as plt


# In[2]:


Feature = {'age':['0','1'],
           'job':["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"],
           'marital':["married","divorced","single"],
           'education':["unknown","secondary","primary","tertiary"],
           'default':["yes","no"],
           'balance':['0','1'],
          'housing':["yes","no"],
          'load':["yes","no"],
          'contact':["unknown","telephone","cellular"],
          'day':['0','1'],
          'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
          'duration':['0','1'],
          'campaign':['0','1'],
          'pdays':['0','1'],
          'previous':['0','1'],
          'poutcome':["unknown","other","failure","success"]}
      
Column = list(Feature.keys())
Label = ['yes','no']


# In[113]:


t_data = []
with open ('processed_train.csv','r') as f:
    for line in f :
        terms = line.strip().split(',')
        t_data.append(terms)


# In[114]:


test_data = []
with open ('processed_test.csv','r') as f:
    for line in f :
        terms = line.strip().split(',')
        test_data.append(terms)


# In[5]:


def judge_label(S): #judge if there are same labels in set S and what the label is.
    first_label = t_data[S[0]][-1]
    l = S.shape[0]
    judge = 1
    for i in range(1,l):
        if t_data[S[i]][-1] != first_label:
            judge = 0
            break
    return first_label,judge


# In[6]:


def find_most_label(S,weight):
    l = S.shape[0]
    count = np.zeros(len(Label))
    for i in range(0,l):
        ind = Label.index(t_data[S[i]][-1])
        count[ind] = count[ind] + weight[S[i]]    ############
    first_label = Label[np.argmax(count)]
    return first_label


# In[7]:


def data_separate(S,a):
    values = Feature[a]
    d = len(values)
    l = S.shape[0]
    p = Column.index(a)
    V = []
    T = []
    for i in range(0,d):
        V.append([])
    for j in range(0,l):
        ind = values.index(t_data[S[j]][p])
        V[ind].append(S[j])
    for i in range(0,d):
        T.append(np.array(V[i]))
    return T
    


# In[8]:


def compute_entropy(S,weight):
    if S.shape[0] == 0:
        return 0
    else:
        l = S.shape[0]
        tw = np.sum(weight[S]) ##
        d = len(Label)
        count = np.zeros(d)
        ii = np.arange(d)
        for i in range(0,l):
            ind = Label.index(t_data[S[i]][-1])
            count[ind] = count[ind] + weight[S[i]] ##
        p = count/tw
        p = np.delete(p,ii[p==0])
        entr = -np.dot(p,np.log(p))
        return entr


# In[9]:


def compute_information_entropy(S,a,weight):
    V = data_separate(S,a)
    
    
    ws = weight[S] #weight of this subset
    tw = np.sum(ws)
    
    
    l = S.shape[0]
    d = len(V)
    entr = 0
    for i in range(0,d):
        entr = entr + np.sum(weight[np.int_(V[i])])/tw * compute_entropy(V[i],weight) ##
    entr = compute_entropy(S,weight) - entr
    return entr


# In[10]:


def best_split(S,A,weight):
    d = len(A)
    entropy = np.zeros(d)
    for i in range(0,d):
        entropy[i] = compute_information_entropy(S,A[i],weight)
    ind = np.argmax(entropy)
    return A[ind]


# In[11]:


def ID3(setS,listA,layer,weight):
    S = copy.deepcopy(setS)
    A = copy.deepcopy(listA)
    if S.shape[0] == 0:
        return -1
    else:
        first_label,judge = judge_label(S) #no need to consider weight
        if judge == 1:
            return first_label
        else:
            if A==[]:
                first_label = find_most_label(S,weight) #need to consider. But note that in Adaboost case with high probability A will not be empty.
                return first_label
            else:
                root = {}
                subtree = {}
                attr = best_split(S,A,weight) #need to consider
                V  = data_separate(S,attr)
                d = len(V)
                if layer - max_layer == 0:
                    for i in range(0,d):
                        if V[i].shape[0] != 0:
                            subtree[Feature[attr][i]] = find_most_label(V[i],weight) #
                        else:
                            subtree[Feature[attr][i]] = find_most_label(S,weight)   #
                else:
                    A.remove(attr)
                    for i in range(0,d):
                        result = ID3(V[i],A,layer+1,weight) 
                        if result != -1:
                            subtree[Feature[attr][i]] = result
                        else:
                            subtree[Feature[attr][i]] = find_most_label(S,weight) #
                root[attr] = subtree
                return root


# In[109]:


def predict(data,Tree):
    if isinstance(Tree,dict):
        f = list(Tree.keys())[0]
        ind = Column.index(f)
        T = Tree[f][data[ind]]
        y = predict(data,T)
    else:
        y = Tree
    return y


# In[90]:


def bagging_predict(data,T):
    l = len(T)
    count = np.zeros(len(Label))
    for i in range(0,l):
        y = predict(data,T[i])
        ind = Label.index(y)
        count[ind] = count[ind] + 1
    dec = np.argmax(count)
    return dec


# In[14]:


def compute_ada_error(data,T,alpha): ##need to adjust
    num = len(data)
    count = 0
    for i in range(0,num):
        p_label = ada_predict(data[i],T,alpha)
        if p_label != data[i][-1]:
            count = count + 1
    error = count/num
    return error


# In[15]:


def compute_tree_error(data,tree,w):
    num = len(data)
    count = 0
    for i in range(0,num):
        p_label = predict(data[i],tree)
        if p_label != data[i][-1]:
            count = count + w[i]
    error = count
    return error


# In[16]:


def update_weight(data,tree,w,alpha):
    num = len(data)
    nw = np.zeros(num)
    for i in range(0,num):
        p_label = predict(data[i],tree)
        if p_label != data[i][-1]:
            nw[i] = w[i] * np.exp(alpha)
        else:
            nw[i] = w[i] * np.exp(-alpha)
    nw = nw/np.sum(nw)
    return nw


# In[83]:


def Bagging(s_size,weight0,newset):
    Tree = []
    error_train = np.zeros(max_iter)
    error_test = np.zeros(max_iter)
    w = weight0
    for i in range(0,max_iter):
        ini_S = np.random.choice(newset,size = s_size)
        tree = ID3(ini_S,ini_A,1,w)
        Tree.append(tree)
        error_train[i] = compute_tree_error(t_data,tree,w)
        error_test[i] = compute_tree_error(test_data,tree,w)
        #print(i)
    return error_train,error_test,Tree


# In[ ]:


num = len(t_data)
ini_A = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'load', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
max_layer = 20
max_iter = 1000
#initialize weight vector
weight0 = np.ones(num)/num
ss = 800
error_train,error_test,Tree = Bagging(ss,weight0)


# In[88]:


num = len(t_data)
ini_A = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'load', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
max_layer = 20
max_iter = 1000
TREE = []
weight0 = np.ones(num)/num
ss = 800
for i in range(0,25):
    ns = np.random.choice(np.arange(num),size = 1000, replace = False)
    error_train,error_test,Tree = Bagging(ss,weight0,ns)
    TREE.append(Tree)
    print(i)


# In[91]:


##for forest predictions computing
num_t = len(test_data)
preds = np.zeros([25,num_t])
for i in range(0,25):
    print(i)
    for j in range(0,num_t):
        preds[i,j] = bagging_predict(test_data[j],TREE[i])


# In[112]:


#compute bias and variance 
y_true = np.array([Label.index(i[-1]) for i in test_data])
p_mean = np.mean(preds,axis=0)
p_var = np.var(preds,axis=0)
g_var = np.mean(p_var)
g_bias = np.mean(np.square(y_true-p_mean))
print(p_var.shape)
print(g_bias)
print(g_var+g_bias)


# In[110]:


##for first tree predictions computing
num_t = len(test_data)
preds_t = np.zeros([25,num_t])
for i in range(0,25):
    print(i)
    for j in range(0,num_t):
        preds_t[i,j] = Label.index(predict(test_data[j],TREE[i][0]))


# In[111]:


p_mean_t = np.mean(preds_t,axis=0)
p_var_t = np.var(preds_t,axis=0)
g_var_t = np.mean(p_var_t)
g_bias_t = np.mean(np.square(y_true-p_mean_t))
print(g_var_t)
print(g_bias_t)
print(g_var_t+g_bias_t)


# In[86]:


Tree = TREE[0]
Error_train = np.zeros(max_iter)
Error_test = np.zeros(max_iter)
num_t = len(test_data)
l = len(Tree)
record = np.zeros([num,len(Label)])
#all prediction should be recorded
for i in range(0,l):
    for j in range(0,num):
        y = predict(t_data[j],Tree[i])
        ind = Label.index(y)
        record[j,ind] = record[j,ind] + 1
        dec = np.argmax(record[j,:])
        if Label[dec] != t_data[j][-1]:
            Error_train[i] = Error_train[i] + 1
Error_train = Error_train / num

for i in range(0,l):
    for j in range(0,num_t):
        y = predict(test_data[j],Tree[i])
        ind = Label.index(y)
        record[j,ind] = record[j,ind] + 1
        dec = np.argmax(record[j,:])
        if Label[dec] != test_data[j][-1]:
            Error_test[i] = Error_test[i] + 1
Error_test = Error_test / num_t


# In[87]:


plt.plot(Error_test,'r',Error_train,'b')


# In[ ]:


print(Error_test)


# In[61]:


plt.plot(error_train,'r')


# In[ ]:




