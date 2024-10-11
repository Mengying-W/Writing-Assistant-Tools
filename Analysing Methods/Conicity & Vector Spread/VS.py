from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

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
    lenth = len(sensveclist)
    return sensveclist,lenth

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
    lenth = len(sensveclist)
    return sensveclist,lenth

def getALBERTvecs(veclist):
    sensvecslist = veclist.split('],') #all sen vec list for every para
    sensveclist = []
    #preprocess: remove strings and transform to float
    for senvec in sensvecslist:
        tem = senvec.replace(' ','').strip('[').strip(']').split(',')
        tem = [float(x) for x in tem]
        #print(tem)
        sensveclist.append(tem)
    lenth = len(sensveclist)
    return sensveclist,lenth

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
    lenth = len(sensveclist)
    return sensveclist,lenth

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
    lenth = len(float_vectors)
    return float_vectors,lenth

df1 = pd.read_csv('../emvcOfgpt2-1.csv')
df1 = df1.drop(df1.columns[0],axis=1)
df2 = pd.read_csv('../emvcOfgpt2-2.csv')
df2 = df2.drop(df2.columns[0],axis=1)
df3 = pd.read_csv('../emvcOfgpt2-3.csv')
df3 = df3.drop(df3.columns[0],axis=1)
df4 = pd.read_csv('../emvcOfgpt2-4.csv')
df4 = df4.drop(df4.columns[0],axis=1)
df5 = pd.read_csv('../emvcOfgpt2-5.csv')
df5 = df5.drop(df5.columns[0],axis=1)
df6 = pd.read_csv('../emvcOfgpt2-6.csv')
df6 = df6.drop(df6.columns[0],axis=1)

df = pd.concat([df1,df2,df3,df4,df5,df6], ignore_index=True, axis=1)
heads = ['Original','Rephrase','Grammarly','ChatGPT','Wordtune','Quillbot','Vicuna','GPT_4','Flan_T5_1','Flan_T5_2','Flan_T5_3','Flan_T5_4','Flan_T5_5','Flan_T5_6','Flan_T5_7','Flan_T5_8','Flan_T5_9','Flan_T5_10']
df.columns = heads

df_allvec = pd.DataFrame()

for head in heads:
    allvectors = df[head].tolist()
    length = len(allvectors)
    
    float_vecs = []
    for i in tqdm(range(length)):
        vectors_tem = allvectors[i]
        #print(vectors)
        vectors = getGPT2vecs(vectors_tem)
        float_vecs.append(vectors)
    df_allvec[head] = float_vecs

df1 = pd.read_csv('../meanvecs_gpt2-1.csv')
df1 = df1.drop(df1.columns[0],axis=1)
df2 = pd.read_csv('../meanvecs_gpt2-2.csv')
df2 = df2.drop(df2.columns[0],axis=1)
df3 = pd.read_csv('../meanvecs_gpt2-3.csv')
df3 = df3.drop(df3.columns[0],axis=1)
df4 = pd.read_csv('../meanvecs_gpt2-4.csv')
df4 = df4.drop(df4.columns[0],axis=1)
df5 = pd.read_csv('../meanvecs_gpt2-5.csv')
df5 = df5.drop(df5.columns[0],axis=1)
df6 = pd.read_csv('../meanvecs_gpt2-6.csv')
df6 = df6.drop(df6.columns[0],axis=1)

df_mean = pd.concat([df1,df2,df3,df4,df5,df6], ignore_index=True, axis=1)
heads = ['Original','Rephrase','Grammarly','ChatGPT','Wordtune','Quillbot','Vicuna','GPT_4','Flan_T5_1','Flan_T5_2','Flan_T5_3','Flan_T5_4','Flan_T5_5','Flan_T5_6','Flan_T5_7','Flan_T5_8','Flan_T5_9','Flan_T5_10']
df_mean.columns = heads

df_mean_vec = pd.DataFrame()
for head in heads:
    digit_value = []
    values = df_mean[head].tolist()
    for indeics in tqdm(range(len(values))):
        value = values[indeics]
        value = value.strip('[').strip(']').split(',')
        #.replace('\n','').replace('  ',' ').replace('   ',' ').split(' ')
        value = [x for x in value if len(x) != 0]
        '''if len(value) != 768:
            value = value[0:768]'''
        value = [float(x) for x in value]
        digit_value.append(value)
    df_mean_vec[head] = digit_value

df_conicity = pd.read_csv('../conicity_gpt2.csv')
df_conicity = df_conicity.drop(df_conicity.columns[0],axis=1)

df_VS = pd.DataFrame()

for head in heads:
    all_vecs = df_allvec[head].tolist()
    meanvecs = df_mean_vec[head].tolist()
    conicities = df_conicity[head].tolist()
    
    all_vector_spread = []
    
    for all_vec,meanvec,conicity in tqdm(zip(all_vecs,meanvecs,conicities)):
        #print(len(all_vec))
        meanvec = np.array(meanvec)
        
        vector_spread_sum = 0.0
        
        for vec in all_vec:
            #print(len(vec))
            vec = np.array(vec)
            atm = np.dot(vec, meanvec) / (np.linalg.norm(vec) * np.linalg.norm(meanvec))
            vector_spread_single = pow((atm - conicity),2)
            vector_spread_sum = vector_spread_sum + vector_spread_single
            
        vector_spread = vector_spread_sum / len(all_vecs)
        all_vector_spread.append(vector_spread)
        
    df_VS[head] = all_vector_spread

df_VS.to_csv('../vector_spread_gpt2.csv')

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
    lenth = len(sensveclist)
    return sensveclist,lenth

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
    lenth = len(sensveclist)
    return sensveclist,lenth

def getALBERTvecs(veclist):
    sensvecslist = veclist.split('],') #all sen vec list for every para
    sensveclist = []
    #preprocess: remove strings and transform to float
    for senvec in sensvecslist:
        tem = senvec.replace(' ','').strip('[').strip(']').split(',')
        tem = [float(x) for x in tem]
        #print(tem)
        sensveclist.append(tem)
    lenth = len(sensveclist)
    return sensveclist,lenth

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
    lenth = len(sensveclist)
    return sensveclist,lenth

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
    lenth = len(float_vectors)
    return float_vectors,lenth
