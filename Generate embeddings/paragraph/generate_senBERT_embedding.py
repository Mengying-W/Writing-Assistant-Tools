#!/usr/bin/env python
# coding: utf-8

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

#!pip install -U sentence-transformers

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

df = pd.read_csv('../annotated data.csv')
heads = df.columns.values.tolist()

sentences = "I love the iron man!"
sentence_embeddings = model.encode(sentences)

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

emvecs.to_csv('../sen2vec_senBERT.csv')
