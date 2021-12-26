import time
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel


def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


top_n = 5
use_title = True
use_abst = True
query_title = "Edge-Level Explanations for Graph Neural Networks by Extending Explainability Methods for Convolutional Neural Networks"
query_abst = "Graph Neural Networks (GNNs) are deep learning models that take graph data as inputs, and they are applied to various tasks such as traffic prediction and molecular property prediction. However, owing to the complexity of the GNNs, it has been difficult to analyze which parts of inputs affect the GNN model's outputs. In this study, we extend explainability methods for Convolutional Neural Networks (CNNs), such as Local Interpretable Model-Agnostic Explanations (LIME), Gradient-Based Saliency Maps, and Gradient-Weighted Class Activation Mapping (Grad-CAM) to GNNs, and predict which edges in the input graphs are important for GNN decisions. The experimental results indicate that the LIME-based approach is the most efficient explainability method for multiple tasks in the real-world situation, outperforming even the state-of-the-art method in GNN explainability."

tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")
model = model.to("cpu")

with open('./data/raw_title.pkl', 'rb') as f:
    title_list = pickle.load(f)
with open('./data/raw_abst.pkl', 'rb') as f:
    abst_list = pickle.load(f)
with open('./data/raw_link.pkl', 'rb') as f:
    link_list = pickle.load(f)

query_input = tokenizer(query_title, return_tensors='pt').to("cpu")
query1 = model(**query_input).pooler_output[0]
query_input = tokenizer(query_abst, return_tensors='pt').to("cpu")
query2 = model(**query_input).pooler_output[0]

title_embeddings = np.load('./data/title_embeddings.npy')
abst_embeddings = np.load('./data/abst_embeddings.npy')

if use_title and use_abst:
    query = torch.cat((query1, query2)).cpu().detach().numpy()
    embeddings = np.concatenate([title_embeddings, abst_embeddings], axis=1)
elif use_title:
    query = query1.cpu().detach().numpy()
    embeddings = title_embeddings
elif use_abst:
    query = query2.cpu().detach().numpy()
    embeddings = abst_embeddings
else:
    raise ValueError

sim_list = []
start_time = time.time()
for vector in embeddings:
    sim_list.append(cos_similarity(query, vector))
        
sim_list = np.array(sim_list)
sim_idx = np.argsort(sim_list)[::-1]
for i in range(top_n):
    print('Similarlity: {}'.format(sim_list[sim_idx[i]]))
    print('Title: {}'.format(title_list[sim_idx[i]]))
    print('Link: {}'.format(link_list[sim_idx[i]]))
    print('Abstract: \n{}'.format(abst_list[sim_idx[i]]))

print(time.time() - start_time)

with open("./data/past-result.txt", mode='w') as f:
    for i in range(top_n):
        f.write('Similarlity: {}\n'.format(sim_list[sim_idx[i]]))
        f.write('Title: {}\n'.format(title_list[sim_idx[i]]))
        f.write('Link: {}\n'.format(link_list[sim_idx[i]]))
        f.write('Abstract: \n{}\n\n'.format(abst_list[sim_idx[i]]))