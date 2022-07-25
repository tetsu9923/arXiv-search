import argparse
import pickle

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


def Dim_reduction(sentences, tokenizer, model):
    vecs = []
    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True,  max_length=64)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            vec = output_hidden_state.cpu().numpy()[0]
            vecs.append(vec)

    kernel, bias = compute_kernel_bias([vecs])
    kernel = kernel[:, :128]
    embeddings = []
    embeddings = np.vstack(vecs)
    embeddings = transform_and_normalize(
        embeddings, 
        kernel=kernel,
        bias=bias
    )
    return embeddings


def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)
    
def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    
def compute_kernel_bias(vecs):
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


def batch(list1, list2, batch_size=1):
    l = len(list1)
    assert l == len(list2)
    for ndx in range(0, l, batch_size):
        yield list1[ndx:min(ndx + batch_size, l)], list2[ndx:min(ndx + batch_size, l)]


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size

    with open('./data/raw_title.pkl', 'rb') as f:
        title_list = pickle.load(f)
    with open('./data/raw_abst.pkl', 'rb') as f:
        abst_list = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter').to(device)

    for i, (title, abst) in enumerate(batch(title_list, abst_list, batch_size=batch_size)):
        with torch.no_grad():
            output = Dim_reduction(title, tokenizer, model)

            if i == 0:
                title_embeddings = output
            else:
                title_embeddings = torch.cat((title_embeddings, output), dim=0)

            del _input
            del output
            torch.cuda.empty_cache()
        print(batch_size*(i+1))

    for i, (title, abst) in enumerate(batch(title_list, abst_list, batch_size=batch_size)):
        with torch.no_grad():
            output = Dim_reduction(abst, tokenizer, model)
    
            if i == 0:
                abst_embeddings = output
            else:
                abst_embeddings = torch.cat((abst_embeddings, output), dim=0)

            del _input
            del output
            torch.cuda.empty_cache()
        print(batch_size*(i+1))

    np.save('./data/title_embeddings.npy', title_embeddings.cpu().detach().numpy())
    np.save('./data/abst_embeddings.npy', abst_embeddings.cpu().detach().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    main(args)