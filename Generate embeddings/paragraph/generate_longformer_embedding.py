#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import LongformerTokenizer, LongformerModel
import pandas as pd
from tqdm import tqdm
import numpy as np

# Load the Longformer tokenizer and model
model = LongformerModel.from_pretrained("allenai/longformer-large-4096")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096")

# Define a function to compute the document embedding vector
def get_document_embedding(document):
    # Tokenize the document and add special tokens
    tokens = tokenizer.encode(document, add_special_tokens=True)
    # Convert the token IDs to a PyTorch tensor
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Batch size 1
    # Compute the document embedding vector using the Longformer model
    outputs = model(input_ids)
    # Extract the output embedding from the last layer
    last_hidden_states = outputs.last_hidden_state
    # Take the mean of the embeddings across all positions
    doc_embedding = torch.mean(last_hidden_states, dim=1).squeeze().tolist()
    return doc_embedding

df = pd.read_csv('../annotated data.csv')

df_vec = pd.DataFrame()
for index in df:
    para = df[index].tolist()
    final_vec = []
    for i in tqdm(range(len(para))):
        document = para[i]
        embedding = get_document_embedding(document)
        final_vec.append(embedding)
    df_vec[index] = final_vec

df_vec.to_csv('../doc2vec_longformer.csv')
