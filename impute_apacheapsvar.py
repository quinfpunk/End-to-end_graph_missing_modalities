import torch
import pandas as pd
import os
import argparse
import numpy as np
from torch_sparse import SparseTensor, mul, sum, fill_diag, matmul, eye, mul_nnz
from src.eicu_dataset import eICUDataset
import scipy.sparse as sp
from eICU_similarity import eICUPatientSimilarity
from tqdm import tqdm

np.random.seed(42)

dataset = eICUDataset(split="train", task="mortality", load_no_label=True)


parser = argparse.ArgumentParser(description="Run imputation.")
# parser.add_argument('--data', type=str, default='Office_Products')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--method', type=str, default='neigh_mean')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--top_k', type=int, default=20)
args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# visual_folder = f'data/visual_embeddings/torch/ResNet50/avgpool'
# textual_folder = f'data/textual_embeddings/sentence_transformers/sentence-transformers/all-mpnet-base-v2/1'

output_apacheapsvar = f'src/apacheapsvar_embeddings_{args.method}'

try:
    missing_apacheapsvar = pd.read_csv(os.path.join(f'src', 'apacheapsvar_flag_filtered_missing.tsv'), sep='\t', header=None)
    missing_apacheapsvar = set(missing_apacheapsvar[1][1:].tolist())
except pd.errors.EmptyDataError:
    missing_apacheapsvar = set()

# print(len(missing_apacheapsvar))

if args.method == 'neigh_mean':
    if not os.path.exists(output_apacheapsvar + f'_{args.top_k}_indexed'):
        os.makedirs(output_apacheapsvar + f'_{args.top_k}_indexed')

if args.method == 'neigh_mean':
    # code_folder with all embedding
    apacheapsvar_folder = f'src/processed_data/embeddings'

    # apacheapsvar_embedding dataframe
    apacheapsvar_embeddings = pd.read_csv(os.path.join(apacheapsvar_folder, 'apacheapsvar_embeddings.tsv'), sep='\t', header=0)

    # shape
    temp = apacheapsvar_embeddings.loc[1]['embedding'].strip()[7:-1]
    temp = eval(f"{temp}")
    apacheapsvar_shape = len(temp)

    output_apacheapsvar = f'src/apacheapsvar_embeddings_{args.method}_{args.top_k}_indexed'

    try:
        train = pd.read_csv(f'src/train_indexed.tsv', sep='\t', header=None)
        train = train.drop(index=0)
        train[1] = train[1].apply(lambda x: int(x))
        train[2] = train[2].apply(lambda x: int(x[7:-2]))
    except FileNotFoundError:
        print('Before imputing through neigh_mean, split the dataset into train/val/test!')
        exit()
    # print(train[1])
    # print(train[1].min())

    num_items_apacheapsvar = len(missing_apacheapsvar) + len(apacheapsvar_embeddings)
    print(apacheapsvar_embeddings[apacheapsvar_embeddings['id'] == 2744154])
    print("num_items_apacheapsvar: ", num_items_apacheapsvar)

    apacheapsvar_features = torch.zeros((num_items_apacheapsvar, apacheapsvar_shape), device=device)

    # adj = get_item_item()

    try:
        missing_apacheapsvar_indexed = pd.read_csv(os.path.join(f'src', 'apacheapsvar_flag_filtered_missing.tsv'), sep='\t',
                                             header=0)
        missing_apacheapsvar_indexed = set(missing_apacheapsvar_indexed['index'].tolist())
    except (pd.errors.EmptyDataError, FileNotFoundError):
        missing_apacheapsvar_indexed = set()

    # print(len(missing_apacheapsvar_indexed))

    for i in range(len(apacheapsvar_embeddings)):
        id = apacheapsvar_embeddings.iloc[i]['id']
        embedding = apacheapsvar_embeddings[apacheapsvar_embeddings['id'] == id]['embedding'].item()  # str
        embedding = embedding.strip()[7:-1]
        embedding = eval(f'{embedding}')  # list
        embedding = torch.tensor(embedding)  # tensor
        apacheapsvar_features[i, :] = embedding
    # print(train[train[0] == 0][1])

    patient_similarity = eICUPatientSimilarity(dataset, train)

    # Similarity matrix
    similarity = patient_similarity.get_similarity()
    print("similarity shape: ", similarity.size())
    for miss in tqdm(missing_apacheapsvar_indexed):
        idx = train[train[1] == miss][0].item()
        topk = 20
        first_hop = torch.topk(similarity[idx], k=topk)
        mean_ = apacheapsvar_features[first_hop].mean(axis=0, keepdims=True)

    apacheapsvar_features = apacheapsvar_features.numpy()
    np.savetxt("apacheapsvar_features", apacheapsvar_features, delimiter="\t", fmt="%.4f")
