#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
from tqdm import tqdm

def get_float_senBERT_vector(vectorstrlist):
    float_vectors = []
    for item in vectorstrlist:
        item = item.replace(" ","").replace("\n","")
        item = item.strip("(").strip(")").strip("[").strip("]")
        item = item.split(',')[0:384]
        item[383] = item[383].strip(']')
        item = [float(x) for x in item]
        #print(item)
        float_vectors.append(item)
    return float_vectors

df_allvectors = pd.read_csv('../sen2vec_senBERT.csv')
df_allvectors = df_allvectors.drop(df_allvectors.columns[0],axis=1)
heads = df_allvectors.columns.values.tolist()

all_central_points = pd.DataFrame()
for head in heads:
    allvectors = df_allvectors[head].tolist()
    length = len(allvectors)
    
    central_points = []
    for i in tqdm(range(length)):
        vectors = allvectors[i]
        vectors = vectors.strip("[").strip("]").split('array')
        vectors = [x for x in vectors if len(x) != 0]
        vectors = get_float_senBERT_vector(vectors)
        #print(vectors)
        # Set the number of clusters
        k = 1

        # Initialize and fit the KMedoids model
        kmedoids = KMedoids(n_clusters=k, random_state=0)
        kmedoids.fit(vectors)
        # Get the cluster medoids
        medoids = kmedoids.cluster_centers_
        central_points.append(medoids)

        #Print the results
        #print("Medoids:\n", medoids)
    all_central_points[head] = central_points

all_central_points.to_csv('../central_points_senBERT.csv')

def get_floatvec_ALBERT(vecslist):
    vecslist = vecslist.split('],')
    float_vecs = []
    for item in vecslist:
        item = item.strip('[').strip(']').replace(' ','')
        item = item.split(',')
        single_values = []
        for vecs in item:
            vecs = vecs.strip('[')
            single_values.append(vecs)
        single_values = [float(x) for x in single_values]
        float_vecs.append(single_values)
    return float_vecs

df_allvectors = pd.read_csv('../sen2vec_ALBERT.csv')
df_allvectors = df_allvectors.drop(df_allvectors.columns[0],axis=1)
heads = df_allvectors.columns.values.tolist()

all_central_points = pd.DataFrame()
for head in heads:
    allvectors = df_allvectors[head].tolist()
    length = len(allvectors)
    
    central_points = []
    for i in tqdm(range(length)):
        vectors = allvectors[i]
        vectors = get_floatvec_ALBERT(vectors)
        #print(vectors)
        # Set the number of clusters
        k = 1
        
        if len(vectors) == 0 :
            print('chucule')
            vectors = [[0.0 for i in range(768)] for j in range(15)]

        # Initialize and fit the KMedoids model
        kmedoids = KMedoids(n_clusters=k, random_state=0)
        kmedoids.fit(vectors)
        # Get the cluster medoids
        medoids = kmedoids.cluster_centers_
        central_points.append(medoids)

        #Print the results
        #print("Medoids:\n", medoids)
    all_central_points[head] = central_points

all_central_points.to_csv('../central_points_ALBERT.csv')
