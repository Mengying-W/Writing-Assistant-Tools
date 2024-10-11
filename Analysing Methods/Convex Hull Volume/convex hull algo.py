# %%
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import os

# %%
def get_convex_hull_volume(sent_embeddings):
    
    lenOfpoints = len(sent_embeddings)
    #if lenOfpoints >= 9:
    pca = PCA(n_components=8)
    '''else :
        numOfdim = lenOfpoints - 1
        pca = PCA(n_components=numOfdim)'''
    cluster_embs = pca.fit_transform(sent_embeddings)
    hull = ConvexHull(cluster_embs)
    return hull.volume



# %%
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

# %%
base = '/home/pop532211/WATs/processed/word'

for root, ds, fs in os.walk(base):
    for f in fs:
        fullpath = os.path.join(root, f)
        print(fullpath)
        df = pd.read_csv(fullpath)
        df = df.drop(df.columns[0],axis=1)
        heads = df.columns.values.tolist()

        all_volume = pd.DataFrame()
        for head in heads:
            allvectors = df[head].tolist()
            length = len(allvectors)

            volumes = []
            for i in tqdm(range(length)):
                vectors = allvectors[i]
                '''vectors = vectors.strip("[").strip("]").split('array')
                numofvec = len(vectors)
                vectors = vectors[1:numofvec]'''
                vectors = getBERTcasevecs(vectors)

                try :
                    volume = get_convex_hull_volume(vectors)
                except :
                    volume = 0.0
                    print(i)
                volumes.append(volume)

            all_volume[head] = volumes

        filename = str(f)[6:]
        resultsfolder = '/home/pop532211/WATs/convexhull volume/word/'

        func = 'volume_'
        filenames = func + filename
        resultspath = os.path.join(resultsfolder, filenames)
        print(resultspath)
        all_volume.to_csv(resultspath)

# %%



