#!/usr/bin/env python
# coding: utf-8

# In[156]:


import numpy as np
import copy


# In[3]:


Feature = {'buying':['vhigh','high','med','low'],
           'maint':['vhigh','high','med','low'],
           'doors':['2', '3', '4', '5more'],
           'persons':['2', '4', 'more'],
           'lug_boot':['small', 'med', 'big'],
           'safety':['low', 'med', 'high']}
Column = ['buying','maint','doors','persons','lug_boot','safety']
Label = ['unacc','acc','good','vgood']


# In[4]:


t_data = []
with open ('train.csv','r') as f:
    for line in f :
        terms = line.strip().split(',')
        t_data.append(terms)


# In[169]:


test_data = []
with open ('test.csv','r') as f:
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


def find_most_label(S):
    l = S.shape[0]
    count = np.zeros(len(Label))
    for i in range(0,l):
        ind = Label.index(t_data[S[i]][-1])
        count[ind] = count[ind] + 1
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
    


# In[252]:


def compute_entropy(S):
    if S.shape[0] == 0:
        return 0
    else:
        l = S.shape[0]
        d = len(Label)
        count = np.zeros(d)
        ii = np.arange(d)
        for i in range(0,l):
            ind = Label.index(t_data[S[i]][-1])
            count[ind] = count[ind] + 1
        p = count/l
        p = np.delete(p,ii[p==0])
        entr = -np.dot(p,np.log(p))
        return entr


# In[9]:


def compute_information_entropy(S,a):
    V = data_separate(S,a)
    l = S.shape[0]
    d = len(V)
    entr = 0
    for i in range(0,d):
        entr = entr + V[i].shape[0]/l * compute_entropy(V[i])
    entr = compute_entropy(S) - entr
    return entr


# In[256]:


def compute_ME(S):
    l = S.shape[0]
    if l == 0:
        return 0
    else:
        count = 0
        m_label = find_most_label(S)
        for i in range(0,l):
            if t_data[S[i]][-1] != m_label:
                count = count+1
        entr = count/l
        return entr


# In[262]:


def compute_majority_error(S,a):
    V = data_separate(S,a)
    l = S.shape[0]
    d = len(V)
    entr = 0
    for i in range(0,d):
        entr = entr + V[i].shape[0]/l * compute_ME(V[i])
    entr = compute_ME(S) - entr
    return entr


# In[278]:


def compute_GINI_index(S,a):
    V = data_separate(S,a)
    l = S.shape[0]
    d = len(V)
    entr = 0
    for i in range(0,d):
        entr = entr + V[i].shape[0]/l * compute_GINI(V[i])
    entr = compute_GINI(S) - entr
    return entr


# In[289]:


def compute_GINI(S):
    l = S.shape[0]
    if l == 0:
        return 0
    else:
        values = Label
        d = len(values)
        V = []
        p = np.zeros(d)
        for i in range(0,d):
            V.append([])
        for j in range(0,l):
            ind = values.index(t_data[S[j]][-1])
            V[ind].append(S[j])
        for i in range(0,d):
            p[i] = len(V[i])/l
        entr = 1 - np.dot(p,p)
        return entr


# In[279]:


def best_split(S,A):
    d = len(A)
    entropy = np.zeros(d)
    for i in range(0,d):
        entropy[i] = compute_information_entropy(S,A[i])
    ind = np.argmax(entropy)
    return A[ind]


# In[283]:


def ID3(setS,listA,layer):
    S = copy.deepcopy(setS)
    A = copy.deepcopy(listA)
    if S.shape[0] == 0:
        return -1
    else:
        first_label,judge = judge_label(S)
        if judge == 1:
            return first_label
        else:
            if A==[]:
                first_label = find_most_label(S)
                return first_label
            else:
                root = {}
                subtree = {}
                attr = best_split(S,A)
                V  = data_separate(S,attr)
                d = len(V)
                if layer - max_layer == 0:
                    for i in range(0,d):
                        if V[i].shape[0] != 0:
                            subtree[Feature[attr][i]] = find_most_label(V[i])
                        else:
                            subtree[Feature[attr][i]] = find_most_label(S)  
                else:
                    A.remove(attr)
                    for i in range(0,d):
                        result = ID3(V[i],A,layer+1)
                        if result != -1:
                            subtree[Feature[attr][i]] = result
                        else:
                            subtree[Feature[attr][i]] = find_most_label(S)
                root[attr] = subtree
                return root


# In[12]:


def predict(data,Tree):
    if isinstance(Tree,dict):
        f = list(Tree.keys())[0]
        ind = Column.index(f)
        T = Tree[f][data[ind]]
        y = predict(data,T)
    else:
        y = Tree
    return y


# In[194]:


def compute_error(data,tree):
    num = len(data)
    count = 0
    for i in range(0,num):
        p_label = predict(data[i],tree)
        if p_label != data[i][-1]:
            count = count + 1
    error = count/num
    return error


# In[299]:


num = len(t_data)
ini_S = np.arange(num)
ini_A = ['buying','maint','doors','persons','lug_boot','safety']
max_layer = 6
tree = ID3(ini_S,ini_A,1)
test_error = compute_error(test_data,tree)
train_error = compute_error(t_data,tree)
print(test_error)
print(train_error)







