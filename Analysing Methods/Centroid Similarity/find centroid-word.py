#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
from tqdm import tqdm


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


# In[5]:


filenames = ['emvcOfbert-uncase2.csv','emvcOfbert-uncase3.csv','emvcOfbert-uncase4.csv','emvcOfbert-uncase5.csv','emvcOfbert-uncase6.csv']


# In[6]:


for filename in filenames:
    full_path = '/home/pop532211/WATs/processed/word/' + filename
    df = pd.read_csv(full_path)
    df = df.drop(df.columns[0],axis=1)
    #df.head()
    heads = df.columns.values.tolist()
    all_central_points = pd.DataFrame()
    for head in heads:
        allvectors = df[head].tolist()
        #allvectors = allvectors[0:2]
        length = len(allvectors)
        
        central_points = []
        for i in tqdm(range(length)):
            vectors_tem = allvectors[i]
            #vectors = getALBERTvecs(vectors_tem)
            #vectors = getLongformervecs(vectors_tem)
            vectors = getBERTcasevecs(vectors_tem)

            # Set the number of clusters
            k = 1

            # Initialize and fit the KMedoids model
            kmedoids = KMedoids(n_clusters=k, random_state=0)
            kmedoids.fit(vectors)
            # Get the cluster medoids
            medoids = kmedoids.cluster_centers_
            medoids = medoids.tolist()
            central_points.append(medoids)

            #Print the results
            #print("Medoids:\n", medoids)
        all_central_points[head] = central_points
    save_path = '/home/pop532211/WATs/central vecs/sentence/word/central_' + filename
    all_central_points.to_csv(save_path)
   


# In[ ]:




