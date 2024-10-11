# %pip install openai

import openai
import pandas as pd
from tqdm import tqdm
import csv

openai.api_key = "my GPT4 API key" # set the model here

def gpt_4(input):

    model = "gpt-4" # set the model
    #prompt = "could you help me to rephrase this paragraph: " + input
    prompt = "Please rephrase the following paragraph. Preserve the meaning but change the words: '\n" + input + "'\n " + "Rephrased paragraph:"
    messages = [
                    #{
                        #"role" : "system", #should I add the system role?
                        #"content" : "you are a writer"
                    #}
                    {
                        "role" : "user",
                        "content" : prompt
                    }
                ]

    rephrased_text = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature=1 #I am not sure how to set this parameter
    )

    return rephrased_text.choices[0].message.content

df = pd.read_csv("../original text.csv")
paragraphs = df['paragraph'].values.tolist()
paragraphs = paragraphs[14:]

all_gpt4_rewritten_text = []
i = 0
for line in tqdm(paragraphs):
    i = i + 1
    original_text = line
    print(i)
    rewritten_text = gpt_4(original_text)
    print(rewritten_text)
    all_gpt4_rewritten_text.append(rewritten_text)

df1 = pd.DataFrame()
df1["answer"] = all_gpt4_rewritten_text
df1.to_csv("../GPT4_output.csv")
