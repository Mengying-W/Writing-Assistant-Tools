# %%
#from transformers import XLNetTokenizer, XLNetModel
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# %%
# Load XLNet tokenizer and model
'''tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased')'''
tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
model = AutoModelForCausalLM.from_pretrained("xlnet-large-cased")

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

df_vec.to_csv('../sen2vec_xlnet.csv')
