#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from tqdm import tqdm
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import random


# # Jaccard Similarity

# In[24]:


import nltk
import nltk.stem
from nltk.corpus import stopwords
import string

import math
from collections import Counter

nltk.download('stopwords')
stop = set(stopwords.words('english'))
tqdm.pandas()


# In[8]:


df_original = pd.read_csv('/Users/carina/Downloads/courses/paper/dataset for the paper/annotated data.csv')


# In[25]:


punctuation_map = dict((ord(char), None) for char in string.punctuation)  #引入标点符号，为下步去除标点做准备
s = nltk.stem.SnowballStemmer('english')   #在提取词干时,语言使用英语,使用的语言是英语


# In[26]:


def stem_count(text):
    l_text = text.lower()     #全部转化为小写以方便处理 
    without_punctuation = l_text.translate(punctuation_map)    #去除文章标点符号
    tokens = nltk.word_tokenize(without_punctuation)        #将文章进行分词处理,将一段话转变成一个list
    without_stopwords = [w for w in tokens if w not in stop]    #去除文章的停用词
    cleaned_text = [] 
    for i in range(len(without_stopwords)):
        cleaned_text.append(s.stem(without_stopwords[i]))    #提取词干
        
    return cleaned_text


# In[27]:


def jaccard_sim(a, b):
    x = stem_count(str(a))
    y = stem_count(str(b))
    unions = len(set(x).union(set(y)))
    intersections = len(set(x).intersection(set(y)))
    return intersections / unions


# In[28]:


df_outputs = df_original.drop(df_original.columns[0],axis=1)#remove original para

#original_keywords = df_keywords['paragraph'].tolist()
original_words = df_original['Original'].tolist()
df_jacc_sim = pd.DataFrame()

heads = df_outputs.columns.values.tolist()
for head in heads:
    jacc_sims = []
    tem_list = df_outputs[head].tolist()
    for i,j in zip(original_words,tem_list):
        jacc_sim = jaccard_sim(i, j)
        jacc_sims.append(jacc_sim)
    df_jacc_sim[head] = jacc_sims


# In[29]:


df_jacc_flant5 = df_jacc_sim
for head in heads:
    if "Flan_T5" not in head:
        df_jacc_flant5 = df_jacc_flant5.drop(head,axis=1)#remove original para
        
row_means = df_jacc_flant5.mean(axis=1)
df_jacc_sim['Flan-T5_Means'] = row_means

df_jacc = df_jacc_sim
for head in heads:
    if "Flan_T5" in head:
        df_jacc = df_jacc.drop(head,axis=1)#remove original para
        
df_jacc_sim.to_csv('/Users/carina/Downloads/courses/paper/processed data/word/df_jacc_sim.csv')


# In[30]:


df_jacc = pd.read_csv('/Users/carina/Downloads/courses/paper/processed data/word/df_jacc_sim.csv')
sorted_x_data = ['Grammarly','Wordtune','Quillbot','Rephrase','ChatGPT','GPT_4','Vicuna','Flan-T5_Means']
plt.figure(figsize=(13,8)) 
sns.boxplot(data=df_jacc,order=sorted_x_data,showfliers=False,palette="Set3",width=0.75, linewidth=1.5) 
plt.legend(loc = 1, bbox_to_anchor = (1,1))

plt.yticks(fontproperties = 'Times New Roman', size = 22)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
plt.savefig('/Users/carina/Downloads/courses/paper/plots/word/Jaccard-Similarity.png')


# # Length Distribution

# In[6]:


def get_numof_words(document):
    if len(str(document)) == 0:
        return 0
    else:
        senten = str(document).split(' ')
        count = len(senten)
        return count


# In[9]:


df_numwords = pd.DataFrame()
for index in df_original:
    para = df_original[index].tolist()
    final_num = []
    for i in tqdm(range(len(para))):
        document = para[i]
        numwords = get_numof_words(document)
        final_num.append(numwords)
    df_numwords[index] = final_num


# In[21]:


df_flant5_length = df_numwords
heads = df_numwords.columns.values.tolist()
for head in heads:
    if "Flan_T5" not in head:
        df_flant5_length = df_flant5_length.drop(head,axis=1)#remove original para
        
row_means = df_flant5_length.mean(axis=1)
df_numwords['Flan-T5_Means'] = row_means

df_len = df_numwords
for head in heads:
    if "Flan_T5" in head:
        df_len = df_len.drop(head,axis=1)#remove original para
        
df_numwords.to_csv('/Users/carina/Downloads/courses/paper/processed data/word/df_length.csv')


# In[23]:


df_len = pd.read_csv('/Users/carina/Downloads/courses/paper/processed data/word/df_length.csv')
sorted_x_data = ['Original','Grammarly','Wordtune','Quillbot','Rephrase','ChatGPT','GPT_4','Vicuna','Flan-T5_Means']
plt.figure(figsize=(13,8)) 
sns.boxplot(data=df_len,order=sorted_x_data,showfliers=False,palette="Set3",width=0.75, linewidth=1.5) 
plt.legend(loc = 1, bbox_to_anchor = (1,1))

plt.yticks(fontproperties = 'Times New Roman', size = 22)
plt.xticks(fontproperties = 'Times New Roman', size = 16)
plt.savefig('/Users/carina/Downloads/courses/paper/plots/word/paragraph-length.png')


# In[ ]:




