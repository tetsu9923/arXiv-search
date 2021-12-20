import time
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel


def batch(list1, list2, batch_size=1):
    l = len(list1)
    assert l == len(list2)
    for ndx in range(0, l, batch_size):
        yield list1[ndx:min(ndx + batch_size, l)], list2[ndx:min(ndx + batch_size, l)]


with open('./data/raw_title.pkl', 'rb') as f:
    title_list = pickle.load(f)
with open('./data/raw_abst.pkl', 'rb') as f:
    abst_list = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

start_time = time.time()
for i, (title, abst) in enumerate(batch(title_list, abst_list, batch_size=5)):
    _input = tokenizer(title, max_length=512, padding=True, truncation=True, return_tensors='pt')
    output1 = model(**_input).pooler_output

    _input = tokenizer(abst, max_length=512, padding=True, truncation=True, return_tensors='pt')
    output2 = model(**_input).pooler_output
    
    if i == 0:
        title_embeddings = output1
        abst_embeddings = output2
    else:
        title_embeddings = torch.cat((title_embeddings, output1), dim=0)
        abst_embeddings = torch.cat((abst_embeddings, output2), dim=0)

print(time.time() - start_time)
np.save('./data/title_embeddings.npy', title_embeddings.cpu().detach().numpy())
np.save('./data/abst_embeddings.npy', abst_embeddings.cpu().detach().numpy())