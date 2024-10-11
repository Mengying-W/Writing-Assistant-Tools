#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[3]:


#!pip install -U sentence-transformers


# In[6]:


model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# In[7]:


df = pd.read_csv('/home/pop532211/WATs/generate_embeddings/annotated data.csv')
heads = df.columns.values.tolist()


# In[8]:


sentences = "I love the iron man!"
sentence_embeddings = model.encode(sentences)

print(len(sentence_embeddings))
print(sentence_embeddings)


# In[9]:


emvecs = pd.DataFrame()
for head in heads:
    textlist = df[head].tolist()
    emvec = []
    for i in tqdm(range(len(textlist))):
        text = textlist[i]
        sentence_embeddings = model.encode(text)
        #print(len(sentence_embeddings))
        emvec.append( sentence_embeddings)
    emvecs[head] = emvec


# In[10]:


emvecs.to_csv('/home/pop532211/WATs/processed/paragraph/sen2vec_senBERT.csv')


# In[ ]:




