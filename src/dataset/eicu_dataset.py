import os

import torch
from torch.utils.data import Dataset

from src.dataset.tokenizer import eICUTokenizer
from src.utils import processed_data_path, read_txt, load_pickle
import pickle
import numpy


class eICUDataset(Dataset, ):
    def __init__(self, split, task, load_no_label=False, dev=False, return_raw=False):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu_all/15000_icu_stay_dict.pkl"))
        included_admission_ids = read_txt(
            os.path.join(processed_data_path, f"eicu_all/task-{task}/{split}_admission_ids.txt"))
        self.no_label_admission_ids = []
        if load_no_label:
            no_label_admission_ids = read_txt(
                os.path.join(processed_data_path, f"eicu_all/task-{task}/no_label_admission_ids.txt"))
            self.no_label_admission_ids = no_label_admission_ids
            included_admission_ids += no_label_admission_ids
        self.included_admission_ids = included_admission_ids
        if dev:
            self.included_admission_ids = self.included_admission_ids[:10000]
        self.return_raw = return_raw
        self.tokenizer = eICUTokenizer()


    def __len__(self):
        return len(self.included_admission_ids)

    def __getitem__(self, index):
        icu_id = self.included_admission_ids[index]
        # icu_stay: eICUData
        icu_stay = self.all_hosp_adm_dict[icu_id]

        age = str(icu_stay.age)
        gender = icu_stay.gender
        ethnicity = icu_stay.ethnicity
        '''
            codes 
        '''
        # return raw compatibility
        # types = []
        # codes = icu_stay.codes
        types = icu_stay.trajectory[0]
        codes = icu_stay.trajectory[1]
        codes_flag = True

        '''
            lab
        '''
        labvectors = icu_stay.labvectors
        labvectors_flag = True
        if labvectors is None:
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
        label = float(getattr(icu_stay, self.task))
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
        return_dict["types"] = types
        return_dict["codes"] = codes
        return_dict["codes_flag"] = codes_flag

        return_dict["labvectors"] = labvectors
        return_dict["labvectors_flag"] = labvectors_flag

        return_dict["apacheapsvar"] = apacheapsvar
        return_dict["apacheapsvar_flag"] = apacheapsvar_flag

        return_dict["label"] = label

        return return_dict


    def get_item_by_id(self, id):
        return self.all_hosp_adm_dict[id]


if __name__ == "__main__":
    dataset = eICUDataset(split="train", task="mortality", load_no_label=False)
    print(len(dataset))
    item = dataset[0]
    print(item["id"])
    print(item["age"])
    print(item["gender"])
    print(item["ethnicity"])
    print(item["labvectors"])
    print(item["codes"])
    print(item["label"])

