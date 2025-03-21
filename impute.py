import os.path

import torch
import torch.nn.functional as F
import argparse
import numpy as np
from torch_sparse import SparseTensor, mul, sum, fill_diag, matmul

from src.dataset.eicu_dataset import eICUDataset
from src.dataset.tokenizer import eICUTokenizer
from src.eICU_similarity import eICUPatientSimilarity
from src.embedding.codes_embedding import CodeEncoder
from snf import compute
from tqdm import tqdm
from src.utils import output_path
import pickle

np.random.seed(42)


def one_hot_encoder(modal):
    modal_int = modal.to(torch.int64)
    num_modal = int(max(modal))+1
    modal_onehot = F.one_hot(modal_int.flatten(), num_classes=num_modal)
    return modal_onehot


def get_patient_patient():
    # all modalities
    # merge imputed and the original data
    # patient_patient = (age_similarity.matrix + gender_similarity.matrix + ethnicity_similarity.matrix
    #                    + lab_similarity.matrix + apacheapsvar_similarity.matrix + codes_similarity.matrix)
    patient_patient = compute.snf([age_similarity.matrix.numpy(), gender_similarity.matrix.numpy(), ethnicity_similarity.matrix.numpy(),
                                   lab_similarity.matrix.numpy(), apacheapsvar_similarity.matrix.numpy(), codes_similarity.matrix.numpy()])

    knn_val, knn_ind = torch.topk(torch.tensor(patient_patient, device=device), args.top_k, dim=-1)
    patients_cols = torch.flatten(knn_ind).to(device)
    ir = torch.tensor(list(range(patient_patient.shape[0])), dtype=torch.int64, device=device)
    patients_rows = torch.repeat_interleave(ir, args.top_k).to(device)
    final_adj = SparseTensor(row=patients_rows,
                             col=patients_cols,
                             value=torch.tensor([1.0] * patients_rows.shape[0], device=device),
                             sparse_sizes=(patient_patient.shape[0], patient_patient.shape[0]))
    return final_adj


def compute_normalized_laplacian(adj, norm, fill_value=0.):
    adj = fill_diag(adj, fill_value=fill_value)
    deg = sum(adj, dim=-1)
    deg_inv_sqrt = deg.pow_(-norm)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


def impute_apacheapsvar(dataset, adj, apacheapsvar_similarity, layers, device=torch.device('cpu')):
    print("Imputation Apacheapsvar starts.")
    num_patients_apacheapsvar = len(dataset)

    apacheapsvar_features = torch.zeros((num_patients_apacheapsvar, len(dataset[0]['apacheapsvar'])))

    missing_apacheapsvar_indexed = set(dataset.all_hosp_adm_dict.keys()) - set(apacheapsvar_similarity.index_table)
    missing_index_id_map = []

    # feat prop on visual features
    index_id_map = []
    for i in tqdm(range(len(dataset))):
        index_id_map.append(dataset[i]['id'])
        if (not torch.any(dataset[i]['apacheapsvar'] == -1)) and dataset[i]['apacheapsvar_flag']:
            apacheapsvar_features[i] = dataset[i]['apacheapsvar']

    non_missing_patients = apacheapsvar_similarity.index_table
    non_missing_index_id_map = []
    propagated_apacheapsvar_features = apacheapsvar_features.clone()
    for item in missing_apacheapsvar_indexed:
        if item in index_id_map:
            missing_index_id_map.append(index_id_map.index(item))
    for item in non_missing_patients:
        if item in index_id_map:
            non_missing_index_id_map.append(index_id_map.index(item))

    for idx in range(layers):
        print(f'[ApacheApsVar] Propagation layer: {idx + 1}')
        propagated_apacheapsvar_features = matmul(adj.to(device), propagated_apacheapsvar_features.to(device))
        # non_missing_items map
        propagated_apacheapsvar_features[non_missing_index_id_map] = apacheapsvar_features[non_missing_index_id_map].to(device)

    imputed_apacheapsvar = torch.zeros((num_patients_apacheapsvar, len(dataset[0]['apacheapsvar'])))
    for i in range(num_patients_apacheapsvar):
        imputed_apacheapsvar[i] = dataset[i]['apacheapsvar']
    for miss in missing_index_id_map:
        # check dataset，replace -1
        for i in range(len(imputed_apacheapsvar[0])):
            if imputed_apacheapsvar[miss][i] == -1:
                imputed_apacheapsvar[miss][i] = propagated_apacheapsvar_features[miss][i]

    print("Imputation Apacheapsvar finish.")

    return imputed_apacheapsvar


def impute_labevents(dataset, adj, lab_similarity, layers, device=torch.device('cpu')):
    print("Imputation Labevents starts.")
    num_patients_labevents = len(dataset)

    labevents_features = torch.zeros((num_patients_labevents, len(dataset[0]['labvectors'])))

    missing_labevents_indexed = set(dataset.all_hosp_adm_dict.keys()) - set(lab_similarity.index_table)
    missing_index_id_map = []

    index_id_map = []
    for i in tqdm(range(len(dataset))):
        index_id_map.append(dataset[i]['id'])
        if dataset[i]['labvectors_flag']:
            labevents_features[i] = torch.tensor(dataset[i]['labvectors'])

    non_missing_patients = lab_similarity.index_table
    non_missing_index_id_map = []
    propagated_labevents_features = labevents_features.clone()
    for item in missing_labevents_indexed:
        if item in index_id_map:
            missing_index_id_map.append(index_id_map.index(item))
    for item in non_missing_patients:
        if item in index_id_map:
            non_missing_index_id_map.append(index_id_map.index(item))

    for idx in range(layers):
        print(f'[LabEvents] Propagation layer: {idx + 1}')
        propagated_labevents_features = matmul(adj.to(device), propagated_labevents_features.to(device))
        # non_missing_items map
        propagated_labevents_features[non_missing_index_id_map] = labevents_features[non_missing_index_id_map].to(
            device)

    imputed_labevents = torch.zeros((num_patients_labevents, len(dataset[0]['labvectors'])))
    for i in range(num_patients_labevents):
        imputed_labevents[i] = torch.tensor(dataset[i]['labvectors'])
    # imputation
    for miss in missing_index_id_map:
        # check dataset，replace -1
        for i in range(len(imputed_labevents[0])):
            imputed_labevents[miss] = propagated_labevents_features[miss]

    print("Imputation Labevents finish.")

    return imputed_labevents


model = CodeEncoder(tokenizer=eICUTokenizer(), embedding_size=128)
def build_dataset(dataset, imputed_apacheapsvar, imputed_labevents):
    age = torch.zeros(size=(len(dataset), 1))
    gender = torch.zeros(size=(len(dataset), 1))
    ethnicity = torch.zeros(size=(len(dataset), 1))
    for i in tqdm(range(len(dataset))):
        age[i] = dataset[i]["age"]
        gender[i] = dataset[i]["gender"]
        ethnicity[i] = dataset[i]["ethnicity"]
    age_onehot = one_hot_encoder(age)
    gender_onehot = one_hot_encoder(gender)
    ethnicity_onehot = one_hot_encoder(ethnicity)
    # torch.tensor([age, gender, ethnicity, codes, apacheapsvar, labvectors, label])
    for i in tqdm(range(len(dataset))):
        t = []
        id = dataset[i]["id"]
        codes_raw = dataset[i]['codes']
        codes_embedded = model.encode(codes_raw)
        t.append(age_onehot[i])
        t.append(gender_onehot[i])
        t.append(ethnicity_onehot[i])
        t.append(codes_embedded)
        t.append(imputed_apacheapsvar[i])
        t.append(imputed_labevents[i])
        t.append(dataset[i]["label"])
        with open(os.path.join(output_path, f"{id}.pkl"), "wb") as f:
            pickle.dump(t, f)


parser = argparse.ArgumentParser(description="Run imputation.")
# parser.add_argument('--data', type=str, default='Office_Products')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--apache_layers', type=int, default=3)
parser.add_argument('--method', type=str, default='feat_prop')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--top_k', type=int, default=20)
args = parser.parse_args()

device = torch.device('cpu')
train_set = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")
val_set = eICUDataset(split="val", task="mortality")
test_set = eICUDataset(split="test", task="mortality")

patient_similarity = eICUPatientSimilarity(train_set)
age_similarity, gender_similarity, ethnicity_similarity = patient_similarity.get_similarity()
apacheapsvar_similarity = patient_similarity.get_apacheapsvar_similarity()
lab_similarity = patient_similarity.get_labevents_similarity()
codes_similarity = patient_similarity.get_codes_similarity()


if args.method == 'feat_prop':
    # age gender ethnicity adjacency matrix
    adj = get_patient_patient()
    # normalize adjacency matrix
    adj = compute_normalized_laplacian(adj, 0.5)

    # impute train, validation, test
    imputed_apacheapsvar_train = impute_apacheapsvar(train_set, adj, apacheapsvar_similarity, layers=args.apache_layers)
    imputed_labevents_train = impute_labevents(train_set, adj, lab_similarity, layers=args.layers)
    imputed_apacheapsvar_val = impute_apacheapsvar(val_set, adj, apacheapsvar_similarity, layers=args.apache_layers)
    imputed_labevents_val = impute_labevents(val_set, adj, lab_similarity, layers=args.layers)
    imputed_apacheapsvar_test = impute_apacheapsvar(test_set, adj, apacheapsvar_similarity, layers=args.apache_layers)
    imputed_labevents_test = impute_labevents(test_set, adj, lab_similarity, layers=args.layers)

    # save the dataset
    build_dataset(train_set, imputed_apacheapsvar_train, imputed_labevents_train)
    build_dataset(val_set, imputed_apacheapsvar_val, imputed_labevents_val)
    build_dataset(test_set, imputed_apacheapsvar_test, imputed_labevents_test)



