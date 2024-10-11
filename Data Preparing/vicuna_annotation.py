#
#!pip install bitsandbytes
#!pip install --no-index --no-deps /kaggle/input/bitsandbytes/bitsandbytes-0.41.1-py3-none-any.whl
#!pip install accelerate
#!pip install scipy

# %%
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import transformers
import bitsandbytes
import pandas as pd
from tqdm import tqdm
import random

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel, AutoModelForCausalLM

# %%
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
    print('Device name:', torch.cuda.get_device_name(1))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# %%
config = {
    'model':{
        'model_checkpoint': 'lmsys/vicuna-13b-v1.5',
    },
    'inference':{
        'num_sequences': 2
    }
}
offload_folder = "/Users/carina/Downloads"

# %%
model = AutoModelForCausalLM.from_pretrained(
    config['model']['model_checkpoint'], 
    device_map="auto", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    offload_folder=offload_folder
    #load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained(config['model']['model_checkpoint'], use_fast=True)

# %%
df = pd.read_csv("/home/pop532211/WATs/WATs-LLMs/original text.csv")
paragraphs = df['paragraph'].values.tolist()
paragraphs = paragraphs[0:10]
'''random.seed(42)
paragraphs = random.sample(paragraphs, 40)'''

# %%
df1 = pd.DataFrame()
df1['original'] = paragraphs

answers_vicuna_13b = []
for item in tqdm(paragraphs):

    input_text = "Please rephrase the following paragraph. Preserve the meaning but change the words: '\n" + item + "'\n " + "Rephrased paragraph:"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    outputs = model.generate(

        input_ids = input_ids.to(device),
            #attention_mask = inputs['attention_mask'].to(device), 
        num_return_sequences=1,

        max_length = 1024)

        #answer = tokenizer.decode(outputs[0])
    output_ids = outputs[0].to(device)
    answer = tokenizer.decode(output_ids)
    #print("input:",input_text)
    print("output:",answer)
    answers_vicuna_13b.append(answer)
df1["answer"] = answers_vicuna_13b


# %%
df1.to_csv('/home/pop532211/WATs/WATs-LLMs/vicuna_output.csv')


