#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm

'''from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids'''
import numpy as np


# In[2]:


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# In[3]:


df = pd.read_csv('/home/pop532211/WATs/generate_embeddings/annotated data.csv')
heads = df.columns.values.tolist()


# In[9]:


emvecs = pd.DataFrame()
numOfsens = pd.DataFrame()
for head in heads:
    textlist = df[head].tolist()
    #print("paragraph: ",textlist)
    emvec = []
    numOfsen = []
    for i in tqdm(range(len(textlist))):
        text = textlist[i]
        sentences = sent_tokenize(text)
        #print("sentences: ",sentences)
        sentence_embeddings = model.encode(sentences)
        embeddings = []
        for em in sentence_embeddings:
            #print("length of sen em:",len(em))
            embeddings.append(em)

        #print("num of sen ems:",len(sentence_embeddings))
        emvec.append(embeddings)
        numOfsen.append(len(sentence_embeddings))
    #print("num of sens:",numOfsen)
    emvecs[head] = emvec
    numOfsens[head] = numOfsen


# In[10]:


emvecs.to_csv('/home/pop532211/WATs/processed/sentence/sen2vec_senBERT.csv')


# In[11]:


numOfsens.to_csv('/home/pop532211/WATs/processed/sentence/sen_num.csv')


# In[ ]:




