import time
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel


def cos_similarity(x, outputs, eps=1e-8):
    sim_list = []
    for y in outputs:
        nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
        ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
        sim_list.append(torch.dot(nx, ny).item())
    return sim_list

def batch(list1, list2, n=1):
    l = len(list1)
    assert l == len(list2)
    for ndx in range(0, l, n):
        yield list1[ndx:min(ndx + n, l)], list2[ndx:min(ndx + n, l)]


top_n = 5
query_title = "ICON: Implicit Clothed humans Obtained from Normals"
query_abst = 'Current methods for learning realistic and animatable 3D clothed avatars need either posed 3D scans or 2D images with carefully controlled user poses. In contrast, our goal is to learn the avatar from only 2D images of people in unconstrained poses. Given a set of images, our method estimates a detailed 3D surface from each image and then combines these into an animatable avatar. Implicit functions are well suited to the first task, as they can capture details like hair or clothes. Current methods, however, are not robust to varied human poses and often produce 3D surfaces with broken or disembodied limbs, missing details, or non-human shapes. The problem is that these methods use global feature encoders that are sensitive to global pose. To address this, we propose ICON ("Implicit Clothed humans Obtained from Normals"), which uses local features, instead. ICON has two main modules, both of which exploit the SMPL(-X) body model. First, ICON infers detailed clothed-human normals (front/back) conditioned on the SMPL(-X) normals. Second, a visibility-aware implicit surface regressor produces an iso-surface of a human occupancy field. Importantly, at inference time, a feedback loop alternates between refining the SMPL(-X) mesh using the inferred clothed normals and then refining the normals. Given multiple reconstructed frames of a subject in varied poses, we use SCANimate to produce an animatable avatar from them. Evaluation on the AGORA and CAPE datasets shows that ICON outperforms the state of the art in reconstruction, even with heavily limited training data. Additionally, it is much more robust to out-of-distribution samples, e.g., in-the-wild poses/images and out-of-frame cropping. ICON takes a step towards robust 3D clothed human reconstruction from in-the-wild images. This enables creating avatars directly from video with personalized and natural pose-dependent cloth deformation.'

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
for i, (title, abst) in enumerate(batch(title_list, abst_list, 1)):
    _input = tokenizer(title, max_length=512, padding=True, truncation=True, return_tensors='pt')
    output1 = model(**_input).pooler_output

    _input = tokenizer(abst, max_length=512, padding=True, truncation=True, return_tensors='pt')
    output2 = model(**_input).pooler_output
    
    output = torch.cat((output1, output2), dim=1)
    print(query)
    print(query.shape)
    print(output)
    print(output.shape)
    sim = cos_similarity(query, output)
    sim_list += sim
        
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
