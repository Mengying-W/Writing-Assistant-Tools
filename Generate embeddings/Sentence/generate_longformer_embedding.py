#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import LongformerTokenizer, LongformerModel
import pandas as pd
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load the Longformer tokenizer and model
model = LongformerModel.from_pretrained("allenai/longformer-large-4096")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096")

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

df = pd.read_csv('../annotated data.csv')

df_vec = pd.DataFrame()
for index in df:
    para = df[index].tolist()
    #para = para[0:1]
    final_vec = []
    for i in tqdm(range(len(para))):
        document = para[i]
        embedding = get_document_embedding(document)
        final_vec.append(embedding)
    df_vec[index] = final_vec
    
df_vec.to_csv('../sen2vec_longformer.csv')
