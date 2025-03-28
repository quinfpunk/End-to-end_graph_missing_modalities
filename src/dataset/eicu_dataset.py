import os

import torch
from scipy.interpolate.dfitpack import types
from torch.utils.data import Dataset

from src.dataset.tokenizer import eICUTokenizer
from src.utils import processed_data_path, read_txt, load_pickle
import pickle
import numpy


def merge_pickles(path: str):
    """
        @brief:
            Merge pickles at path
    """
    res = {}
    for file in os.listdir(path):
        if ".pkl" in file:
            # open file and load pickle update the dictionary
            patient_id = file.split(".")[0]
            tmp = {}
            with open(file, "rb") as f:
                embeddings = pickle.load(f)
                # the hard coded index are coming from imput.py build dataset function
                tmp[patient_id] = {'age': embeddings[0],
                                   'gender': embeddings[1],
                                   'ethnicity': embeddings[2],
                                   'codes': embeddings[3],
                                   'apacheapsvar': embeddings[4],
                                   'lab': embeddings[5],
                                   'label': embeddings[6]
                                   }
                res.update(tmp)
    return res


class eICUDataset(Dataset, ):
    def __init__(self, split, task, load_no_label=False, dev=False, return_raw=False, data_size=""):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        # merge all {id}.pkl in the folder
        self.all_hosp_adm_dict = merge_pickles(os.path.join(processed_data_path, split))
        # if data_size == "small":
        #     self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu/small_icu_stay_dict.pkl"))
        # elif data_size == "big":
        #     self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu/big_icu_stay_dict.pkl"))
        # else:
        #     self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu/icu_stay_dict.pkl"))
        included_admission_ids = read_txt(
            os.path.join(processed_data_path, f"eicu/task-{task}/{split}_admission_ids.txt"))
        self.no_label_admission_ids = []
        if load_no_label:
            no_label_admission_ids = read_txt(
                os.path.join(processed_data_path, f"eicu/task-{task}/no_label_admission_ids.txt"))
            self.no_label_admission_ids = no_label_admission_ids
            included_admission_ids += no_label_admission_ids
        self.included_admission_ids = included_admission_ids
        if dev:
            self.included_admission_ids = self.included_admission_ids[:10000]
        self.return_raw = return_raw
        self.tokenizer = eICUTokenizer()

    def __len__(self):
        return len(self.all_hosp_adm_dict)

    def __getitem__(self, index):
        keys = list(self.all_hosp_adm_dict.keys())
        icu_id = keys[index]
        # icu_stay: eICUData
        icu_stay = self.all_hosp_adm_dict[icu_id]

        age = str(icu_stay.age)
        gender = icu_stay.gender
        ethnicity = icu_stay.ethnicity
        '''
            codes
        '''
        # return raw compatibility
        types = []
        codes = icu_stay.codes
        codes_flag = True

        '''
            lab
        '''
        labvectors = icu_stay.lab
        labvectors_flag = True
        if len(labvectors) == 0:
            labvectors = numpy.zeros(384)
            labvectors_flag = False

        '''
            apacheapsvar
        '''
        apacheapsvar = icu_stay.apacheapsvar
        apacheapsvar_flag = True
        if apacheapsvar is None:
            apacheapsvar = -torch.zeros(36)
            apacheapsvar_flag = False
        else:
            apacheapsvar = torch.FloatTensor(apacheapsvar)

        '''
            label
        '''
        label = float(icu_stay.label)
        label_flag = True
        if icu_id in self.no_label_admission_ids:
            label_flag = False

        if not self.return_raw:
            age, gender, ethnicity, types_coded, codes = self.tokenizer(
                age, gender, ethnicity, types, codes
            )
            label = torch.tensor(label)

        return_dict = dict()
        return_dict["id"] = icu_id

        return_dict["age"] = age
        return_dict["gender"] = gender
        return_dict["ethnicity"] = ethnicity
        return_dict["codes"] = codes

        return_dict["lab"] = labvectors

        return_dict["apacheapsvar"] = apacheapsvar

        '''
            treatment, medication, diagnosis
        '''
        return_dict["label"] = label

        return return_dict

    def get_item_by_id(self, id):
        return self.all_hosp_adm_dict[id]


if __name__ == "__main__":
    dataset = eICUDataset(split="train", task="mortality", load_no_label=True)
    print(len(dataset))
    item = dataset[0]
    print(item["id"])
    print(item["age"])
    print(item["gender"])
    print(item["ethnicity"])
    print(item["lab"])
    print(item["codes"])
