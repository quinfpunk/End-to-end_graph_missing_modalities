import os

import torch
from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.data import Dataset

from src.dataset.tokenizer import eICUTokenizer
from src.utils import processed_data_path, read_txt, load_pickle


class eICUDataset(Dataset, ):
    def __init__(self, split, task, load_no_label=False, dev=False, return_raw=False, data_size=""):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        if data_size == "small":
            self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu/small_icu_stay_dict.pkl"))
        elif data_size == "big":
            self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu/big_icu_stay_dict.pkl"))
        else:
            self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu/icu_stay_dict.pkl"))
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
        types = icu_stay.trajectory[0]
        codes = icu_stay.trajectory[1]
        codes_flag = True
        # if len(codes) == 0:
        #     codes_flag = False

        '''
        timestamps
        '''
        times = icu_stay.trajectory[2]

        lab = icu_stay.lab
        labvectors = icu_stay.labvectors
        labvectors_flag = True
        # if labvectors is None:
        if len(lab) == 0:
            # if labvectors is none, set to 0
            labvectors = numpy.zeros(384)
            labvectors_flag = False

        apacheapsvar = icu_stay.apacheapsvar
        apacheapsvar_flag = True
        if apacheapsvar is None:
            # if apacheapsvar is none, set to -1
            apacheapsvar = -torch.ones(36)
            apacheapsvar_flag = False
        else:
            apacheapsvar = torch.FloatTensor(apacheapsvar)

        label = float(getattr(icu_stay, self.task))
        label_flag = True
        if icu_id in self.no_label_admission_ids:
            label_flag = False

        if not self.return_raw:
            age, gender, ethnicity, types, codes = self.tokenizer(
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
        return_dict["times"] = times

        return_dict["lab"] = lab
        return_dict["labvectors"] = labvectors
        return_dict["labvectors_flag"] = labvectors_flag

        return_dict["apacheapsvar"] = apacheapsvar
        return_dict["apacheapsvar_flag"] = apacheapsvar_flag

        return_dict["label"] = label
        return_dict["label_flag"] = label_flag

        return return_dict

    def get_item_by_id(self, id):
        '''
        :param id: patient id
        :return: icu_stay
        '''
        return self.all_hosp_adm_dict[id]


if __name__ == "__main__":
    # test
    dataset = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="small")
    print(len(dataset))
    item = dataset[0]
    print(item["id"])
    print(item["age"])
    print(item["gender"])
    print(item["ethnicity"])
    print(item["apacheapsvar"])
    print(item["labvectors"])
    print(item["codes"])
