#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import copy


# In[9]:


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


# In[10]:


t_data = []
with open ('processed_train.csv','r') as f:
    for line in f :
        terms = line.strip().split(',')
        t_data.append(terms)


# In[11]:


test_data = []
with open ('processed_test.csv','r') as f:
    for line in f :
        terms = line.strip().split(',')
        test_data.append(terms)


# In[12]:


def judge_label(S): #judge if there are same labels in set S and what the label is.
    first_label = t_data[S[0]][-1]
    l = S.shape[0]
    judge = 1
    for i in range(1,l):
        if t_data[S[i]][-1] != first_label:
            judge = 0
            break
    return first_label,judge


# In[13]:


def find_most_label(S):
    l = S.shape[0]
    count = np.zeros(len(Label))
    for i in range(0,l):
        ind = Label.index(t_data[S[i]][-1])
        count[ind] = count[ind] + 1
    first_label = Label[np.argmax(count)]
    return first_label


# In[14]:


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
    


# In[15]:


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


# In[16]:


def compute_information_entropy(S,a):
    V = data_separate(S,a)
    l = S.shape[0]
    d = len(V)
    entr = 0
    for i in range(0,d):
        entr = entr + V[i].shape[0]/l * compute_entropy(V[i])
    entr = compute_entropy(S) - entr
    return entr


# In[17]:


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


# In[18]:


def compute_majority_error(S,a):
    V = data_separate(S,a)
    l = S.shape[0]
    d = len(V)
    entr = 0
    for i in range(0,d):
        entr = entr + V[i].shape[0]/l * compute_ME(V[i])
    entr = compute_ME(S) - entr
    return entr


# In[19]:


def compute_GINI_index(S,a):
    V = data_separate(S,a)
    l = S.shape[0]
    d = len(V)
    entr = 0
    for i in range(0,d):
        entr = entr + V[i].shape[0]/l * compute_GINI(V[i])
    entr = compute_GINI(S) - entr
    return entr


# In[20]:


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


# In[89]:


def best_split(S,A):
    d = len(A)
    entropy = np.zeros(d)
    for i in range(0,d):
        entropy[i] = compute_information_entropy(S,A[i])
    ind = np.argmax(entropy)
    return A[ind]


# In[86]:


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


# In[23]:


def predict(data,Tree):
    if isinstance(Tree,dict):
        f = list(Tree.keys())[0]
        ind = Column.index(f)
        T = Tree[f][data[ind]]
        y = predict(data,T)
    else:
        y = Tree
    return y


# In[24]:


def compute_error(data,tree):
    num = len(data)
    count = 0
    for i in range(0,num):
        p_label = predict(data[i],tree)
        if p_label != data[i][-1]:
            count = count + 1
    error = count/num
    return error


# In[113]:


num = len(t_data)
ini_S = np.arange(num)
ini_A = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'load', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
max_layer = 1
tree = ID3(ini_S,ini_A,1)
train_error = compute_error(t_data,tree)
test_error = compute_error(test_data,tree)
print(train_error)
print(test_error)



def getTreeDepth(tree):
    maxDepth = 0
    firstFeat = list(tree.keys())[0]
    secondDict = tree[firstFeat]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth 


print(getTreeDepth)




