import time
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel


def cos_similarity(x, y, eps=1e-8):
    nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
    return torch.dot(nx, ny)    


top_n = 30
query_title = "Survey Paper Generation"
query_abst = "A survey paper is an important resource for researchers to survey a new research field. However, writing a survey paper requires in-depth knowledge of the field and a great deal of time and effort. In this study, we propose a method to automatically generate survey papers."

tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")

with open('./data/raw_title.pkl', 'rb') as f:
    title_list = pickle.load(f)
with open('./data/raw_abst.pkl', 'rb') as f:
    abst_list = pickle.load(f)

query_input = tokenizer(query_title, return_tensors='pt')
query1 = model(**query_input).pooler_output[0]
query_input = tokenizer(query_abst, return_tensors='pt')
query2 = model(**query_input).pooler_output[0]
query = torch.cat((query1, query2))

sim_list = []

max_sim = 0
max_idx = 0
start_time = time.time()
for i, (title, abst) in enumerate(zip(title_list, abst_list)):
    _input = tokenizer(title, max_length=512, truncation=True, return_tensors='pt')
    output1 = model(**_input).pooler_output[0]

    _input = tokenizer(abst, max_length=512, truncation=True, return_tensors='pt')
    output2 = model(**_input).pooler_output[0]
    
    output = torch.cat((output1, output2))
    sim = cos_similarity(query, output2).item()
    print(sim)
    sim_list.append(sim)
    if max_sim < sim:
        max_idx = i
        max_sim = sim
        print(title_list[max_idx])
        
sim_list = np.array(sim_list)
sim_idx = np.argsort(sim_list)[::-1]
for i in range(top_n):
    print("Similarlity: {}".format(sim_list[sim_idx[i]]))
    print(title_list[sim_idx[i]])
    print(abst_list[sim_idx[i]])

print(time.time() - start_time)

with open("/home/t-kasanishi/arXiv-bot/data/past-result.txt", mode='w') as f:
    for i in range(top_n):
        f.write("Similarlity: {}".format(sim_list[sim_idx[i]]))
        f.write(title_list[sim_idx[i]])
        f.write(abst_list[sim_idx[i]])
