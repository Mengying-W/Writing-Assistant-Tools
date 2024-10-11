#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm
import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-large-cased")

df_original = pd.read_csv('../annotated data6.csv')

heads = df_original.columns.values.tolist()
df_emvecOfaw = pd.DataFrame()
for head in heads:
    paras = []
    paras = df_original[head].tolist()
    
    words_embeddings = []
    for i in tqdm(range(len(paras))):
        
        paragraphs = paras[i]
        
        # Tokenize the paragraph and convert the tokens to IDs
        tokens = tokenizer.tokenize(paragraphs)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #print('num of tokens : ')
        #print(len(tokens))
        
        # Convert the input IDs to a PyTorch tensor
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Batch size 1

        # Generate embeddings for the input IDs using the BERT model
        outputs =model(input_ids, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1][0]
        
        word_embeddings = []
        for token in tokens:
            index = tokens.index(token)
            # Extract the corresponding embedding vector from the BERT model output
            embedding = hidden_state[index].detach().numpy()
            embedding = embedding.tolist()
            word_embeddings.append(embedding)
            
        words_embeddings.append(word_embeddings)
        
    df_emvecOfaw[head] = words_embeddings

df_emvecOfaw.to_csv('../emvcOfbert-case6.csv')
