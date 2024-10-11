#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
from tqdm import tqdm

df = pd.read_csv('../emvcOfgpt2-6.csv')
df = df.drop(df.columns[0],axis=1)
#df.head()
heads = df.columns.values.tolist()

all_average_points = pd.DataFrame()
for head in heads:
    allvectors = df[head].tolist()
    #allvectors = allvectors[0:2]
    length = len(allvectors)

    average_points = []
    for i in tqdm(range(length)):
        vectors_tem = allvectors[i]
        #vectors = getALBERTvecs(vectors_tem)
        #vectors = getLongformervecs(vectors_tem)
        #vectors = getxlnetvecs(vectors_tem)
        #vectors = getsenBERTvecs(vectors_tem)
        #vectors = getGPT2vecs(vectors_tem)
        vectors = getBERTcasevecs(vectors_tem)

        vectors_numpy = np.array(vectors)

        average_vector = np.mean(vectors_numpy, axis=0)
        average_vector = average_vector.tolist()
        '''print(len(average_vector))
        print(average_vector)'''

        average_points.append(average_vector)

    print(len(average_points))
    all_average_points[head] = average_points
all_average_points.to_csv('../meanvecs_gpt2-6.csv')

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
