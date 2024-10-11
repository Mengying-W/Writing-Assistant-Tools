#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from tqdm import tqdm


# In[95]:


df_central = pd.read_csv('/home/pop532211/WATs/meanvecs/word/meanvecs_gpt2-6.csv')
df_central = df_central.drop(df_central.columns[0],axis=1)
heads = df_central.columns.values.tolist()

df_central_vec = pd.DataFrame()
for head in heads:
    digit_value = []
    values = df_central[head].tolist()
    for indeics in tqdm(range(len(values))):
        value = values[indeics]
        value = value.strip('[').strip(']').split(',')
        #.replace('\n','').replace('  ',' ').replace('   ',' ').split(' ')
        value = [x for x in value if len(x) != 0]
        '''if len(value) != 768:
            value = value[0:768]'''
        value = [float(x) for x in value]
        digit_value.append(value)
    df_central_vec[head] = digit_value


# In[96]:


df_central_all = pd.read_csv('/home/pop532211/WATs/processed/word/emvcOfgpt2-6.csv')
df_central_all = df_central_all.drop(df_central_all.columns[0],axis=1)
heads = df_central_all.columns.values.tolist()

df_allvec = pd.DataFrame()

for head in heads:
    allvectors = df_central_all[head].tolist()
    length = len(allvectors)
    
    float_vecs = []
    for i in tqdm(range(length)):
        vectors_tem = allvectors[i]
        #print(vectors)
        vectors = getGPT2vecs(vectors_tem)
        float_vecs.append(vectors)
    df_allvec[head] = float_vecs


# In[97]:


df_cos = pd.DataFrame()

for head in heads:
    all_vecs = df_allvec[head].tolist()
    medoidvecs = df_central_vec[head].tolist()
    
    avg_cos = []
    
    for all_vec,medoidvec in tqdm(zip(all_vecs,medoidvecs)):
        #print(len(all_vec))
        medoidvec = np.array(medoidvec)
        
        cos_simi = 0.0
        
        for vec in all_vec:
            #print(len(vec))
            vec = np.array(vec)
            cosine_sim = np.dot(vec, medoidvec) / (np.linalg.norm(vec) * np.linalg.norm(medoidvec))
            cos_simi = cos_simi + cosine_sim
            
        avg_cos_value = cos_simi / len(all_vecs)
        avg_cos.append(avg_cos_value)
        
    df_cos[head] = avg_cos


# In[98]:


df_cos.to_csv('/home/pop532211/WATs/conicity/word/conicity_gpt2-6.csv')


# In[100]:


gpt21 = pd.read_csv('/home/pop532211/WATs/conicity/word/conicity_gpt2-1.csv')
gpt21 = gpt21.drop(gpt21.columns[0],axis=1)
gpt22 = pd.read_csv('/home/pop532211/WATs/conicity/word/conicity_gpt2-2.csv')
gpt22 = gpt22.drop(gpt22.columns[0],axis=1)
gpt23 = pd.read_csv('/home/pop532211/WATs/conicity/word/conicity_gpt2-3.csv')
gpt23 = gpt23.drop(gpt23.columns[0],axis=1)
gpt24 = pd.read_csv('/home/pop532211/WATs/conicity/word/conicity_gpt2-4.csv')
gpt24 = gpt24.drop(gpt24.columns[0],axis=1)
gpt25 = pd.read_csv('/home/pop532211/WATs/conicity/word/conicity_gpt2-5.csv')
gpt25 = gpt25.drop(gpt25.columns[0],axis=1)
gpt26 = pd.read_csv('/home/pop532211/WATs/conicity/word/conicity_gpt2-6.csv')
gpt26 = gpt26.drop(gpt26.columns[0],axis=1)
merged_df = pd.concat([gpt21, gpt22, gpt23,gpt24,gpt25,gpt26], ignore_index=True, axis=1)
heads = ['Original','Rephrase','Grammarly','ChatGPT','Wordtune','Quillbot','Vicuna','GPT_4','Flan_T5_1','Flan_T5_2','Flan_T5_3','Flan_T5_4','Flan_T5_5','Flan_T5_6','Flan_T5_7','Flan_T5_8','Flan_T5_9','Flan_T5_10']
merged_df.columns = heads
merged_df.to_csv('/home/pop532211/WATs/conicity/word/conicity_gpt2.csv')


# In[2]:


def getBERTcasevecs(veclist):

    sensvecslist = veclist.split('],') #all sen vec list for every para
    #print(len(sensvecslist))
    sensveclist = []
    #preprocess: remove strings and transform to float
    for senvec in sensvecslist:
        tem = senvec.replace(' ','').strip('[').strip(']').split(',')
        tem = [float(x) for x in tem]
        #print(len(tem))
        #print(tem)
        sensveclist.append(tem)
    return sensveclist


# In[3]:


def getxlnetvecs(veclist):

    sensvecslist = veclist.split('],') #all sen vec list for every para

    sensveclist = []
    #preprocess: remove strings and transform to float
    for senvec in sensvecslist:
        tem = senvec.replace(' ','').strip('[').strip(']').split(',')
        tem = [float(x) for x in tem]
        '''print(len(tem))
        print(tem)'''
        sensveclist.append(tem)
    return sensveclist


# In[4]:


def getLongformervecs(veclist):

    sensvecslist = veclist.split('],') #all sen vec list for every para

    sensveclist = []
    #preprocess: remove strings and transform to float
    for senvec in sensvecslist:
        tem = senvec.replace(' ','').strip('[').strip(']').split(',')
        tem = [float(x) for x in tem]
        '''print(len(tem))
        print(tem)'''
        sensveclist.append(tem)
    return sensveclist


# In[5]:


def getsenBERTvecs(veclist):
    sensvecslist = veclist.strip("[").strip("]").split('array')
    sensvecslist = [x for x in sensvecslist if len(x) != 0]
    float_vectors = []
    for item in sensvecslist:
        item = item.replace(" ","").replace("\n","")
        item = item.strip("(").strip(")").strip("[").strip("]")
        item = item.split(',')[0:768]
        item[767] = item[767].strip(']')
        
        #print(item)
        item = [float(x) for x in item]
        float_vectors.append(item)
    return float_vectors


# In[6]:


def getALBERTvecs(veclist):
    sensvecslist = veclist.split('],') #all sen vec list for every para
    sensveclist = []
    #preprocess: remove strings and transform to float
    for senvec in sensvecslist:
        tem = senvec.replace(' ','').strip('[').strip(']').split(',')
        tem = [float(x) for x in tem]
        #print(tem)
        sensveclist.append(tem)
    return sensveclist


# In[7]:


def getGPT2vecs(veclist):
    sensvecslist = veclist.split('],') #all sen vec list for every para
    sensveclist = []
    #preprocess: remove strings and transform to float
    for senvec in sensvecslist:
        tem = senvec.replace(' ','').strip('[').strip(']').split(',')
        tem = [float(x) for x in tem]
        #print(tem)
        sensveclist.append(tem)
    return sensveclist


# In[ ]:




