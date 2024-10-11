#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing


# In[2]:


df = pd.read_csv('/Users/carina/Downloads/courses/paper/central vecs/word/central_emvcOfbert-case.csv')
df = df.drop(df.columns[0],axis=1)
#df.head()
heads = df.columns.values.tolist()


# In[3]:


a = df['Original'].tolist()
b = a[0]
b


# In[4]:


df_vec = pd.DataFrame()
heads = df.columns.values.tolist()    # 列名称
for index in heads:
    digit_value = []
    values = df[index].tolist()
    for item in values:
        item = item.replace("\n","").strip('[').strip(']').split(',')
        item = [x for x in item if len(x) != 0]
        item = [float(x) for x in item]
        digit_value.append(item)
    df_vec[index] = digit_value


# In[5]:


paragraph = df_vec['Original'].tolist()
paragraph = np.array(paragraph)


# In[6]:


df_cos = pd.DataFrame()
df_euc = pd.DataFrame()
df_euc1 = pd.DataFrame()

for index in heads:
    cos = []
    euc = []
    digit_value = df_vec[index].tolist()
    digit_value = np.array(digit_value)
    for i,j in zip(paragraph,digit_value):
        cosine_sim = np.dot(i, j) / (np.linalg.norm(i) * np.linalg.norm(j))
        cos.append(cosine_sim)

        euclidean_dist = np.linalg.norm(i - j)
        #print(i-j)
        #print(np.linalg.norm(i - j))
        euc.append(euclidean_dist)
    euc1 = (euc - np.mean(euc)) / np.std(euc)
    #print(euc)
    #euc = preprocessing.normalize(euc, norm='l2')
    df_cos[index] = cos
    df_euc[index] = euc
    df_euc1[index] = euc1
df_cos = df_cos.drop(df_cos.columns[0],axis=1)
df_euc = df_euc.drop(df_euc.columns[0],axis=1)
df_euc1 = df_euc1.drop(df_euc1.columns[0],axis=1)


# In[7]:


df_cos.to_csv('/Users/carina/Downloads/courses/paper/central similarity/word/emvcOfbert-case_cos.csv')
df_euc1.to_csv('/Users/carina/Downloads/courses/paper/central similarity/word/emvcOfbert-case_euc.csv')
df_euc.to_csv('/Users/carina/Downloads/courses/paper/central similarity/word/emvcOfbert-case_unmorlized_euc.csv')


# In[ ]:




