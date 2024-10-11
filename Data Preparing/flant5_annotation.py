#pip install accelerate

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import pandas as pd
import random
from tqdm import tqdm

tokenizer_flant5 = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model_flant5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

# Connect to GPU
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

df = pd.read_csv("../original text.csv")
paragraphs = df['paragraph'].values.tolist() 
#paragraphs = paragraphs[0:5]
'''random.seed(10)
paragraphs = random.sample(paragraphs, 4)'''

print("new test")
df1 = pd.DataFrame()
df1['original'] = paragraphs

for i in range(10):
    answers_flant5 = []
    for paragraph in tqdm(paragraphs):
        tem = paragraph.split(" ")
        length = len(tem)
        #print(length)
        min_lens = int(length*0.9)
        max_lens = int(length*1.1)
        #print(lens)
        input_text = "Please rephrase the following paragraph. Preserve the meaning but change the words: '\n" + paragraph + "'\n " + "Rephrased paragraph:"
        
       
        inputs_ids = tokenizer_flant5(input_text,return_tensors = "pt",).input_ids
        inputs_ids = inputs_ids.to(device)
        
        #print("length of input id:",len(inputs_ids[0]))
        
        outputs = model_flant5.generate(inputs_ids,min_length=min_lens,max_length = max_lens, temperature = 1.6, do_sample = True,top_k=25)
        #outputs = model_flant5.generate(inputs_ids,max_length = 2048, temperature = 1.6, do_sample = True,top_k=25)

        output_ids = outputs[0].to(device)
        
        answer = tokenizer_flant5.decode(output_ids, skip_special_tokens=True)
        answers_flant5.append(answer)
        
    colomu_name = "answer" + str(i)
    df1[colomu_name] = answers_flant5
    
df1.to_csv('../flant5_output.csv')
