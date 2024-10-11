#!/usr/bin/env python
# coding: utf-8

# In[42]:


#!pip install torch
#!pip install --upgrade transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print(torch.cuda.is_available())


# In[43]:


import pandas as pd
from tqdm import tqdm


# In[44]:


gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2-xl", output_hidden_states=True)


# In[45]:


gpt2_model.to("cuda")


# In[82]:


df_original = pd.read_csv('/home/pop532211/WATs/generate_embeddings/annotated data6.csv')


# In[83]:


heads = df_original.columns.values.tolist()
df_emvecOfaw = pd.DataFrame()
for head in heads:
    paras = []
    paras = df_original[head].tolist()
    
    words_embeddings = []
    for i in tqdm(range(len(paras))):
        
        paragraphs = paras[i]
        
        # 使用 tokenizer 将文本转换为 tokens
        tokens = gpt2_tokenizer(paragraphs, return_tensors="pt").to("cuda")
        
        # 使用 model 获取隐藏状态（hidden states）
        with torch.no_grad():
            outputs = gpt2_model(**tokens)
            
        # 获取所有隐藏状态
        #hidden_states = outputs.hidden_states.to("cuda")
        hidden_states = outputs[0].to("cuda")
        
        # 选择最后一层的隐藏状态（可以根据需要选择其他层）
        last_layer_hidden_states = hidden_states
        
        word_embeddings = []
        hidden_size = gpt2_model.config.hidden_size
        for i, token in enumerate(tokens["input_ids"].squeeze().tolist()):
            embedding_vector = last_layer_hidden_states[0, i,:hidden_size].cpu().numpy()
            embedding_vector = embedding_vector.tolist()
            word_embeddings.append(embedding_vector)
                
        words_embeddings.append(word_embeddings)
    df_emvecOfaw[head] = words_embeddings


# In[ ]:


df_emvecOfaw.to_csv('/home/pop532211/WATs/generate_embeddings/emvcOfgpt2-6.csv')


# In[70]:


a = df_emvecOfaw["Quillbot"].tolist()


# In[72]:


len(a[0][0])


# In[ ]:




