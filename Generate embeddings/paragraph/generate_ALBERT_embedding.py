# %%
import torch
from transformers import AlbertTokenizer, AlbertModel
import pandas as pd
from tqdm import tqdm

# Load model directly
#from transformers import AutoTokenizer, AutoModelForMaskedLM

# %%
# Load the pre-trained ALBERT model
tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
model = AlbertModel.from_pretrained('albert-xxlarge-v2')
'''tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2")
model = AutoModelForMaskedLM.from_pretrained("albert-xxlarge-v2")'''

# %%
df = pd.read_csv('/home/pop532211/WATs/generate_embeddings/annotated data.csv')
heads = df.columns.values.tolist()

# %%
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

# %%
emvecs = pd.DataFrame()
for head in heads:
    textlist = df[head].tolist()
    emvec = []
    for i in tqdm(range(len(textlist))):
        text = textlist[i]
        embedding = get_document_embedding(text)
        emvec.append(embedding)
    emvecs[head] = emvec

# %%
emvecs.to_csv('/home/pop532211/WATs/processed/paragraph/sen2vec_ALBERT.csv')

# %%



