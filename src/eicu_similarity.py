import torch
from src.dataset.eicu_dataset import eICUDataset
from tqdm import tqdm
import torch.nn.functional as F


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


class Similarity:
    def __init__(self, matrix, index_table):
        self.matrix = matrix
        self.index_table = index_table


class eICUPatientSimilarity:
    def __init__(self, dataset):
        self.dataset = dataset

    # age, gender, ethnicity
    def get_similarity(self):
        age = torch.zeros(size=(len(self.dataset), 1))
        gender = torch.zeros(size=(len(self.dataset), 1))
        ethnicity = torch.zeros(size=(len(self.dataset), 1))

        print("Calculating similarity matrix......")
        index = list(self.dataset.all_hosp_adm_dict.keys())
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

        age_similarity = Similarity(age_similarity_matrix, index)
        gender_similarity = Similarity(gender_similarity_matrix, index)
        ethnicity_similarity = Similarity(ethnicity_similarity_matrix, index)

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
        apacheapsvar_similarity = Similarity(apacheapsvar_similarity_matrix, index_table)

        return apacheapsvar_similarity


if __name__ == '__main__':
    # test
    dataset = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")
    print(len(dataset))
    print(len(dataset.all_hosp_adm_dict.keys()))

    patient_similarity = eICUPatientSimilarity(dataset)

    age_similarity, gender_similarity, ethnicity_similarity = patient_similarity.get_similarity()
    apacheapsvar_similarity = patient_similarity.get_apacheapsvar_similarity()
    missing_apacheapsvar = set(dataset.all_hosp_adm_dict.keys()) - set(apacheapsvar_similarity.index_table)
    print(type(missing_apacheapsvar))

    print(age_similarity.matrix)
    print(gender_similarity.matrix)
    print(ethnicity_similarity.matrix)
    print(apacheapsvar_similarity.matrix)
