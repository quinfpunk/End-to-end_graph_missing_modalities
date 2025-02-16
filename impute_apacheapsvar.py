import torch
import argparse
import numpy as np
from torch_sparse import SparseTensor, mul, sum, fill_diag, matmul
from src.dataset.eicu_dataset import eICUDataset
from src.eICU_similarity import eICUPatientSimilarity
from tqdm import tqdm

np.random.seed(42)


def get_patient_patient():
    # all modalities
    # merge imputed and the original data
    patient_patient = age_similarity.matrix + gender_similarity.matrix + ethnicity_similarity.matrix

    knn_val, knn_ind = torch.topk(torch.tensor(patient_patient, device=device), args.top_k, dim=-1)
    items_cols = torch.flatten(knn_ind).to(device)
    ir = torch.tensor(list(range(patient_patient.shape[0])), dtype=torch.int64, device=device)
    items_rows = torch.repeat_interleave(ir, args.top_k).to(device)
    final_adj = SparseTensor(row=items_rows,
                             col=items_cols,
                             value=torch.tensor([1.0] * items_rows.shape[0], device=device),
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


parser = argparse.ArgumentParser(description="Run imputation.")
# parser.add_argument('--data', type=str, default='Office_Products')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--method', type=str, default='feat_prop')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--top_k', type=int, default=20)
args = parser.parse_args()

device = torch.device('cpu')
output_apacheapsvar = f'src/apacheapsvar_embeddings_{args.method}'
dataset = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")

patient_similarity = eICUPatientSimilarity(dataset)
age_similarity, gender_similarity, ethnicity_similarity = patient_similarity.get_similarity()
apacheapsvar_similarity = patient_similarity.get_apacheapsvar_similarity()

# if args.method == 'feat_prop':
#     if not os.path.exists(output_apacheapsvar + f'_{args.top_k}_indexed'):
#         os.makedirs(output_apacheapsvar + f'_{args.top_k}_indexed')


if args.method == 'feat_prop':
    # apacheapsvar_folder = f'data/{args.data}/visual_embeddings_indexed'
    #
    # apacheapsvar_shape = np.load(os.path.join(apacheapsvar_folder, os.listdir(apacheapsvar_folder)[0])).shape
    #
    # output_apacheapsvar = f'data/{args.data}/apacheapsvar_embeddings_{args.method}_{args.layers}_{args.top_k}_indexed'
    #
    # try:
    #     train = pd.read_csv(f'data/{args.data}/train_indexed.tsv', sep='\t', header=None)
    # except FileNotFoundError:
    #     print('Before imputing through feat_prop, split the dataset into train/val/test!')
    #     exit()

    num_items_apacheapsvar = len(dataset)

    apacheapsvar_features = torch.zeros((num_items_apacheapsvar, len(dataset[0]['apacheapsvar'])))

    # age gender ethnicity adjacency matrix
    adj = get_patient_patient()

    # normalize adjacency matrix
    adj = compute_normalized_laplacian(adj, 0.5)

    # try:
    #     missing_apacheapsvar_indexed = pd.read_csv(os.path.join(f'data/{args.data}', 'missing_visual_indexed.tsv'), sep='\t',
    #                                          header=None)
    #     missing_apacheapsvar_indexed = set(missing_apacheapsvar_indexed[0].tolist())
    # except (pd.errors.EmptyDataError, FileNotFoundError):
    #     missing_apacheapsvar_indexed = set()
    missing_apacheapsvar_indexed = set(dataset.all_hosp_adm_dict.keys()) - set(apacheapsvar_similarity.index_table)
    missing_index_id_map = []

    # feat prop on visual features
    # for f in os.listdir(f'data/{args.data}/visual_embeddings_indexed'):
    #     apacheapsvar_features[int(f.split('.npy')[0]), :] = torch.from_numpy(
    #         np.load(os.path.join(f'data/{args.data}/visual_embeddings_indexed', f)))
    index_id_map = []
    for i in tqdm(range(len(dataset))):
        index_id_map.append(dataset[i]['id'])
        if (not torch.any(dataset[i]['apacheapsvar'] == -1)) and dataset[i]['apacheapsvar_flag']:
            apacheapsvar_features[i] = dataset[i]['apacheapsvar']

    # non_missing_items = list(set(list(range(num_items_apacheapsvar))).difference(missing_apacheapsvar_indexed))
    non_missing_items = apacheapsvar_similarity.index_table
    non_missing_index_id_map = []
    propagated_apacheapsvar_features = apacheapsvar_features.clone()
    for item in missing_apacheapsvar_indexed:
        if item in index_id_map:
            missing_index_id_map.append(index_id_map.index(item))
    for item in non_missing_items:
        if item in index_id_map:
            non_missing_index_id_map.append(index_id_map.index(item))

    for idx in range(args.layers):
        print(f'[ApacheApsVar] Propagation layer: {idx + 1}')
        propagated_apacheapsvar_features = matmul(adj.to(device), propagated_apacheapsvar_features.to(device))
        # non_missing_items map
        propagated_apacheapsvar_features[non_missing_index_id_map] = apacheapsvar_features[non_missing_index_id_map].to(device)

    # save results
    # for miss in missing_apacheapsvar_indexed:
    #     np.save(os.path.join(output_apacheapsvar, f'{miss}.npy'), propagated_apacheapsvar_features[miss].detach().cpu().numpy())
    imputed_apacheapsvar = torch.zeros((num_items_apacheapsvar, len(dataset[0]['apacheapsvar'])))
    for i in range(num_items_apacheapsvar):
        imputed_apacheapsvar[i] = dataset[i]['apacheapsvar']
    for miss in missing_index_id_map:
        # check dataset，replace -1
        for i in range(len(imputed_apacheapsvar[0])):
            if imputed_apacheapsvar[miss][i] == -1:
                imputed_apacheapsvar[miss][i] = propagated_apacheapsvar_features[miss][i]

    # print(imputed_apacheapsvar.shape)
    print(dataset[0]['apacheapsvar'])
    print(imputed_apacheapsvar[0])
