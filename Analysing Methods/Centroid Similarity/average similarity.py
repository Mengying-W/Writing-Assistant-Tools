#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from tqdm import tqdm

df_central = pd.read_csv('../central_sen_points_xlnet.csv')
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

df_central_all = pd.read_csv('../sen2vec_xlnet.csv')
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
        vectors = getxlnetvecs(vectors_tem)
        float_vecs.append(vectors)
    df_allvec[head] = float_vecs

df_cos = pd.DataFrame()
df_euc = pd.DataFrame()
df_euc1 = pd.DataFrame()

for head in heads:
    all_vecs = df_allvec[head].tolist()
    medoidvecs = df_central_vec[head].tolist()
    
    avg_euc = []
    avg_cos = []
    avg_euc = []
    
    for all_vec,medoidvec in tqdm(zip(all_vecs,medoidvecs)):
        #print(len(all_vec))
        medoidvec = np.array(medoidvec)
        
        euc_dis = 0.0
        cos_simi = 0.0
        euc_dis = 0.0  
        
        for vec in all_vec:
            #print(len(vec))
            vec = np.array(vec)
            cosine_sim = np.dot(vec, medoidvec) / (np.linalg.norm(vec) * np.linalg.norm(medoidvec))
            cos_simi = cos_simi + cosine_sim
            euclidean_dist = np.linalg.norm(vec - medoidvec)
            euc_dis = euc_dis + euclidean_dist
            
        avg_cos_value = cos_simi / (len(all_vecs) - 1)
        avg_cos.append(avg_cos_value)
        
        avg_euc_value = euc_dis / (len(all_vecs) - 1)
        avg_euc.append(avg_euc_value)

    euc1 = (avg_euc - np.mean(avg_euc)) / np.std(avg_euc)
    
    df_cos[head] = avg_cos
    df_euc[head] = avg_euc
    df_euc1[head] = euc1

df_cos.to_csv('../avg_sen_cos_xlnet.csv')
df_euc.to_csv('../avg_sen_euc_xlnet.csv')
df_euc1.to_csv('../avg_sen_euc1_xlnet.csv')
