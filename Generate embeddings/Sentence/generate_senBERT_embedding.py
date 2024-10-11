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

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

df = pd.read_csv('../annotated data.csv')
heads = df.columns.values.tolist()

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

emvecs.to_csv('../sen2vec_senBERT.csv')

numOfsens.to_csv('../sen_num.csv')
