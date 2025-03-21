import torch
from torch_sparse.matmul import spspmm_add

from src.dataset.eicu_dataset import eICUDataset
from tqdm import tqdm
import torch.nn.functional as F

from src.dataset.tokenizer import eICUTokenizer
from src.embedding.codes_embedding import CodeEncoder
from sentence_transformer_embedding import embed_lab
import numpy as np


def one_hot_encoder(modal):
    modal_int = modal.to(torch.int64)
    num_modal = int(max(modal))+1
    modal_onehot = F.one_hot(modal_int.flatten(), num_classes=num_modal)
    return modal_onehot


def euclidean_dist(x, y):
    b = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(b, b)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(b, b).t()
    dist = xx + yy - 2 * torch.mm(x, y.t())
    return dist


def gaussian_kernel(source, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
    n = source.size(0)
    L2_distance = euclidean_dist(source, source)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n ** 2 - n)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    kernel_sum = sum(kernel_val) / len(kernel_val)

    return kernel_sum


def process_codes(codes, times):
    data = []
    t = times[0]
    temp = (t, [])
    for i in range(len(times)):
        if t == times[i]:
            temp[1].append(codes[i])
        else:
            data.append(temp)
            t = times[i]
            temp = (t, [])
            temp[1].append(codes[i])
    data.append(temp)
    return data


def add_zeros(matrix, index, index_table):
    temp_matrix = torch.zeros((len(index), len(index)))
    index_map = {val: i for i, val in enumerate(index_table)}

    for i, idx_i in enumerate(index):
        for j, idx_j in enumerate(index):
            if idx_i in index_map and idx_j in index_map:
                temp_matrix[i, j] = matrix[index_map[idx_i], index_map[idx_j]]

    return temp_matrix


class Similarity:
    def __init__(self, matrix, index_table):
        self.matrix = matrix
        self.index_table = index_table


class eICUPatientSimilarity:
    def __init__(self, dataset):
        self.dataset = dataset
        if type(dataset) == dict:
            self.index = list(self.dataset["id"])
        else:
            self.index = list(self.dataset.all_hosp_adm_dict.keys())

    # age, gender, ethnicity
    def get_similarity(self):
        age = torch.zeros(size=(len(self.dataset), 1))
        gender = torch.zeros(size=(len(self.dataset), 1))
        ethnicity = torch.zeros(size=(len(self.dataset), 1))

        print("Calculating similarity matrix......")
        # index = list(self.dataset.all_hosp_adm_dict.keys())
        for i in tqdm(range(len(self.dataset))):
            age[i] = self.dataset[i]['age']
            gender[i] = self.dataset[i]['gender']
            ethnicity[i] = self.dataset[i]['ethnicity']

        age_onehot = one_hot_encoder(age)
        gender_onehot = one_hot_encoder(gender)
        ethnicity_onehot = one_hot_encoder(ethnicity)
        # print(ethnicity_onehot.shape)

        age_similarity_matrix = gaussian_kernel(age_onehot)
        gender_similarity_matrix = gaussian_kernel(gender_onehot)
        ethnicity_similarity_matrix = gaussian_kernel(ethnicity_onehot)

        age_similarity = Similarity(age_similarity_matrix, self.index)
        gender_similarity = Similarity(gender_similarity_matrix, self.index)
        ethnicity_similarity = Similarity(ethnicity_similarity_matrix, self.index)

        return (age_similarity,
                gender_similarity,
                ethnicity_similarity)

    # apacheapsvar
    def get_apacheapsvar_similarity(self):
        apacheapsvar = []
        index_table = []
        for i in tqdm(range(len(self.dataset))):
            if (not torch.any(self.dataset[i]['apacheapsvar'] == -1)) and self.dataset[i]['apacheapsvar_flag']:
                index_table.append(self.dataset[i]['id'])
                apacheapsvar.append(self.dataset[i]['apacheapsvar'].tolist())

        apacheapsvar = torch.tensor(apacheapsvar)
        apacheapsvar_similarity_matrix = gaussian_kernel(apacheapsvar)
        apacheapsvar_similarity_matrix = add_zeros(apacheapsvar_similarity_matrix, self.index, index_table)

        apacheapsvar_similarity = Similarity(apacheapsvar_similarity_matrix, index_table)

        return apacheapsvar_similarity

    # lab
    def get_labevents_similarity(self):
        lab = []
        index_table = []
        window_size = 60  # group data points that occur within the same
        for i in tqdm(range(len(self.dataset))):
            if self.dataset[i]['labvectors_flag']:
                index_table.append(self.dataset[i]["id"])
                # d = self.dataset[i]["lab"]
                # lab_embedding = embed_lab(d, window_size)
                # lab.append(lab_embedding[0])
                lab.append(self.dataset[i]["labvectors"])

        lab = torch.tensor(lab)
        lab_similarity_matrix = gaussian_kernel(lab)
        lab_similarity_matrix = add_zeros(lab_similarity_matrix, self.index, index_table)
        lab_similarity = Similarity(lab_similarity_matrix, index_table)

        return lab_similarity

    # codes: medication, diagnosis, treatment
    def get_codes_similarity(self):
        codes = []
        index_table = []
        model = CodeEncoder(tokenizer=eICUTokenizer(), embedding_size=128)
        for i in tqdm(range(len(self.dataset))):
            if self.dataset[i]['codes_flag']:
                index_table.append(self.dataset[i]['id'])
                # codes_raw = self.dataset[i]['codes'][1:-1].tolist()
                codes_raw = self.dataset[i]['codes']
                codes_embedded = model.encode(codes_raw).tolist()
                # times = self.dataset[i]['times']
                # codes_processed = process_codes(codes_raw, times)
                # codes_embedded = generate_codes_embedding(codes_processed).detach().tolist()
                codes.append(codes_embedded)

        codes = torch.tensor(codes)

        codes_similarity_matrix = gaussian_kernel(codes)
        # codes_similarity_matrix = add_zeros(codes_similarity_matrix, self.index, index_table)
        codes_similarity = Similarity(codes_similarity_matrix, index_table)

        return codes_similarity


if __name__ == '__main__':
    # test
    dataset = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")
    print(len(dataset))
    print(len(dataset.all_hosp_adm_dict.keys()))

    patient_similarity = eICUPatientSimilarity(dataset)

    age_similarity, gender_similarity, ethnicity_similarity = patient_similarity.get_similarity()
    apacheapsvar_similarity = patient_similarity.get_apacheapsvar_similarity()
    lab_similarity = patient_similarity.get_labevents_similarity()
    codes_similarity = patient_similarity.get_codes_similarity()
    # missing_apacheapsvar = set(dataset.all_hosp_adm_dict.keys()) - set(apacheapsvar_similarity.index_table)
    # print(type(missing_apacheapsvar))

    print(age_similarity.matrix)
    print(gender_similarity.matrix)
    print(ethnicity_similarity.matrix)
    print(apacheapsvar_similarity.matrix)
    print(lab_similarity.matrix)
    print(codes_similarity.matrix)

