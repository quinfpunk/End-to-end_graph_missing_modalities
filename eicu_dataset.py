import os

import torch
from scipy.interpolate.dfitpack import types
from torch.utils.data import Dataset

from src.tokenizer import eICUTokenizer
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
        '''
            typed_coded
        '''
        types_coded = torch.zeros(len(types))

        codes = icu_stay.trajectory[1]
        codes_flag = True

        lab = icu_stay.lab
        labvectors = icu_stay.labvectors
        labvectors_flag = True
        if labvectors is None:
            labvectors = torch.zeros(1, 158)
            labvectors_flag = False
        else:
            labvectors = torch.FloatTensor(labvectors)

        apacheapsvar = icu_stay.apacheapsvar
        apacheapsvar_flag = True
        if apacheapsvar is None:
            apacheapsvar = torch.zeros(36)
            apacheapsvar_flag = False
        else:
            apacheapsvar = torch.FloatTensor(apacheapsvar)

        '''
            treatment, medication, diagnosis
        '''
        if len(icu_stay.treatment) == 0:
            treatment = torch.tensor([])
            treatment_flag = False
        else:
            treatment = [t[2] for t in icu_stay.treatment]
            treatment_flag = True

        if len(icu_stay.medication) == 0:
            medication = torch.tensor([])
            medication_flag = False
        else:
            medication = [m[2] for m in icu_stay.medication]
            medication_flag = True

        if len(icu_stay.diagnosis) == 0:
            diagnosis = torch.tensor([])
            diagnosis_flag = False
        else:
            diagnosis = [d[2] for d in icu_stay.diagnosis]
            diagnosis_flag = True

        label = float(getattr(icu_stay, self.task))
        label_flag = True
        if icu_id in self.no_label_admission_ids:
            label_flag = False

        # if not self.return_raw:
        #     age, gender, ethnicity, types, codes = self.tokenizer(
        #         age, gender, ethnicity, types, codes
        #     )
        #     label = torch.tensor(label)
        # useless code, maybe ? 
        if not self.return_raw:
            '''
                treatment, medication, diagnosis
            '''
            if treatment_flag:
                _1, _2, _3, _4, treatment = self.tokenizer(
                    age, gender, ethnicity, types, treatment
                )
            if medication_flag:
                _1, _2, _3, _4, medication = self.tokenizer(
                    age, gender, ethnicity, types, medication
                )
            if diagnosis_flag:
                _1, _2, _3, _4, diagnosis = self.tokenizer(
                    age, gender, ethnicity, types, diagnosis
                )
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
        return_dict["types_coded"] = types_coded
        return_dict["codes"] = codes
        return_dict["codes_flag"] = codes_flag

        return_dict["lab"] = lab
        return_dict["labvectors"] = labvectors
        return_dict["labvectors_flag"] = labvectors_flag

        return_dict["apacheapsvar"] = apacheapsvar
        return_dict["apacheapsvar_flag"] = apacheapsvar_flag

        '''
            treatment, medication, diagnosis
        '''
        return_dict["treatment"] = treatment
        return_dict["treatment_flag"] = treatment_flag

        return_dict["medication"] = medication
        return_dict["medication_flag"] = medication_flag

        return_dict["diagnosis"] = diagnosis
        return_dict["diagnosis_flag"] = diagnosis_flag

        return_dict["label"] = label
        return_dict["label_flag"] = label_flag

        return return_dict

    def get_item_by_id(self, id):
        return self.all_hosp_adm_dict[id]


if __name__ == "__main__":
    # dataset = eICUDataset(split="train", task="mortality", load_no_label=True, return_raw=True)
    # print(len(dataset))
    # item = dataset[0]
    # print(item["id"])
    # print(item["age"])
    # print(item["gender"])
    # print(item["ethnicity"])
    # print(len(item["types"]))
    # print(len(item["codes"]))
    # print(item["labvectors"].shape)
    # print(item["apacheapsvar"].shape)
    # print(item["label"])

    from torch.utils.data import DataLoader

    from dataset.utils import eicu_collate_fn

    dataset = eICUDataset(split="train", task="mortality", load_no_label=True)
    print(len(dataset))
    item = dataset[0]
    print(item["id"])
    print(item["age"])
    print(item["gender"])
    print(item["ethnicity"])
    # print(item["types"].shape)
    # print(item["codes"].shape)
    # print(item["label"].shape)

    ### modified code
    import pandas as pd
    from tqdm import tqdm

    training = {}
    training["id"] = []
    training["age"] = []
    # create a dataframe with a column for each feature that could be missing (a "_flag" column exist in the dataset
    present_values = pd.DataFrame()
    for elt in tqdm(dataset):
        training["id"].append(elt["id"])
        training["age"].append(elt["age"])

        # elt_df = pd.DataFrame(elt)
        # for each {column_name}_flag if 0 adds to missing with "id": ["column_name_of_missing", ...]
        # for present_flag in elt_df.filter(regex="_flag$", axis=1):
        for present_flag in elt.keys():
            # for this id add the present flag to False (this means it is missing)
            if "_flag" in present_flag:
                present_values.loc[elt["id"], present_flag] = elt[present_flag]
    # just to be safe ?
    present_values = present_values.fillna(False)

    # save train_indexed
    training_df = pd.DataFrame(training)
    # maybe the path to tsv should be defined using the processed_path in utils ?
    print(training_df)
    # training_df.to_csv("train_indexed.tsv", sep='\t')

    # # save present features, one tsv per feature
    # for col in present_values.columns:
    #     # check that the tsv is not ill formed
    #     filtered_missing_values = present_values.loc[present_values[col] == False]
    #     filtered_missing_values = filtered_missing_values.reset_index()[['index']]
    #     filtered_missing_values.to_csv(f"{col}_filtered_missing.tsv", sep='\t')

    # data_loader = DataLoader(dataset, batch_size=32, collate_fn=eicu_collate_fn, shuffle=True)
    # batch = next(iter(data_loader))
    # print(batch["age"])
    # print(batch["gender"])
    # print(batch["ethnicity"])
    # print(batch["types"].shape)
    # print(batch["codes"].shape)
    # print(batch["codes_flag"])
    # print(batch["labvectors"].shape)
    # print(batch["labvectors_flag"])
    # print(batch["apacheapsvar"])
    # print(batch["apacheapsvar_flag"])
    # print(batch["label"])
    # print(batch["label_flag"])