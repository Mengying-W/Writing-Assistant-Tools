#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
from transformers import AlbertTokenizer, AlbertModel
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import numpy as np

# Load model directly
#from transformers import AutoTokenizer, AutoModelForMaskedLM


# In[14]:


# Define a function to compute the document embedding vector
def get_document_embedding(document):
    sentences = sent_tokenize(document)
    embeddings = []
    #print("sentences: ",sentences)
    for sentence in sentences:
        # 使用tokenizer将文本转换为XLNet的输入格式
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        # 获取XLNet模型的输出
        with torch.no_grad():
            outputs = model(**inputs,output_hidden_states=True)

        # 获取嵌入向量
        embedding_vector = outputs.hidden_states[-1].mean(dim=1).squeeze().tolist()

        embeddings.append(embedding_vector)
        
    return embeddings


# In[15]:


# Load the pre-trained ALBERT model
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
model = AlbertModel.from_pretrained('albert-xxlarge-v2')
'''tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2")
model = AutoModelForMaskedLM.from_pretrained("albert-xxlarge-v2")'''


# In[16]:


# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[17]:


print(device)


# In[18]:


df = pd.read_csv('/home/pop532211/WATs/generate_embeddings/annotated data.csv')
heads = df.columns.values.tolist()


# In[19]:


# Define a function to compute the document embedding vector
def get_document_embedding(document):
    sentences = sent_tokenize(document)
    embeddings = []
    #print("sentences: ",sentences)
    for sentence in sentences:
        # Tokenize the document and add special tokens
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        # Convert the token IDs to a PyTorch tensor
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)  # Batch size 1
        # Compute the document embedding vector using the Longformer model
        outputs = model(input_ids)
        # Extract the output embedding from the last layer
        last_hidden_states = outputs.last_hidden_state
        # Take the mean of the embeddings across all positions
        doc_embedding = torch.mean(last_hidden_states, dim=1).squeeze().tolist()

        embeddings.append(doc_embedding)
        
    return embeddings


# In[21]:


emvecs = pd.DataFrame()
for head in heads:
    textlist = df[head].tolist()
    #textlist = textlist[0:1]
    emvec = []
    for i in tqdm(range(len(textlist))):
        text = textlist[i]
        embedding = get_document_embedding(text)
        emvec.append(embedding)
    emvecs[head] = emvec


# In[22]:


emvecs.to_csv('/home/pop532211/WATs/processed/sentence/sen2vec_ALBERT.csv')


# In[ ]:




