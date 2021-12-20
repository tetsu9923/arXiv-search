import datetime
import torch
import pickle
from transformers import AutoTokenizer, AutoModel


def cos_similarity(x, y, eps=1e-8):
    nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
    return torch.dot(nx, ny)    

query_title = "Edge-Level Explanations for Graph Neural Networks by Extending Explainability Methods for Convolutional Neural Networks"
query_abst = "Graph Neural Networks (GNNs) are deep learning models that take graph data as inputs, and they are applied to various tasks such as traffic prediction and molecular property prediction. However, owing to the complexity of the GNNs, it has been difficult to analyze which parts of inputs affect the GNN model's outputs. In this study, we extend explainability methods for Convolutional Neural Networks (CNNs), such as Local Interpretable Model-Agnostic Explanations (LIME), Gradient-Based Saliency Maps, and Gradient-Weighted Class Activation Mapping (Grad-CAM) to GNNs, and predict which edges in the input graphs are important for GNN decisions. The experimental results indicate that the LIME-based approach is the most efficient explainability method for multiple tasks in the real-world situation, outperforming even the state-of-the-art method in GNN explainability."

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

with open('./data/raw_title.pkl', 'rb') as f:
    title_list = pickle.load(f)
with open('./data/raw_abst.pkl', 'rb') as f:
    abst_list = pickle.load(f)

query_input = tokenizer(query_title+" "+query_abst, return_tensors='pt')
query = model(**query_input).pooler_output[0]

sim_list = []

max_sim = 0
max_idx = 0
for i, (title, abst) in enumerate(zip(title_list, abst_list)):
    _input = title+" "+abst
    if len(_input.split(" ")) > 500:
        _input = " ".join(_input.split(" ")[:500])
        print(_input)
    _input = tokenizer(_input, return_tensors='pt')
    output = model(**_input).pooler_output[0]
    sim = cos_similarity(query, output)
    print(sim)
    sim_list.append(sim)
    if max_sim < sim:
        max_idx = i
        max_sim = sim
        
print(title_list[max_idx])
print(abst_list[max_idx])

with open("/home/t-kasanishi/arXiv-bot/data/daily-result.txt", mode='a') as f:
    f.write(datetime.date.today().strftime("%Y-%m-%d") + "\n")
    f.write(title_list[max_idx] + "\n")
    f.write(abst_list[max_idx] + "\n\n")