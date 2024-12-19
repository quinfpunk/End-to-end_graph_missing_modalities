from src.eicu_dataset import eICUDataset
import torch

import os
import csv


def save_tsv(dict, path, file_name):
    headers = ['id', 'embeddings']
    file_path = os.path.join(path, file_name)
    with open(file=file_path, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        for id, embeddings in dict.items():
            writer.writerow({'id': id, 'embeddings': embeddings})
    print("Saved ", file_name)


if __name__ == '__main__':
    train_set = eICUDataset(split="train", task='mortality', dev=False, load_no_label=False)
    val_set = eICUDataset(split="val", task='mortality')
    test_set = eICUDataset(split="test", task='mortality')
    '''
    type:
        train_set: eICUDataset, consists of many eICUData
            eICUData: similar to a dict
        train_set.__getitem__(0): dict
            Keys: id, 
                  age, gender, ethnicity, 
                  types, 
                  types_coded, 
                  codes, 
                  codes_flag: True, ?
                  labvectors, 
                  labvectors_flag: False if labvectors is missing, else True, 
                  apacheapsvar, 
                  apacheapsvar_flag: False if apacheapsvar is missing, else True, 
                  treatment, 
                  treatment_flag, 
                  medication, 
                  medication_flag, 
                  diagnosis, 
                  diagnosis_flag, 
                  label, 
                  label_flag: False if label is missing, else True
    '''
    treatment_embeddings = {}
    medication_embeddings = {}
    diagnosis_embeddings = {}
    for i in range(train_set.__len__()):
        dict_i = train_set.__getitem__(i)
        types = dict_i["types"]
        id = dict_i["id"]
        codes = dict_i["codes"][1:-1]

        # If the ID doesn't exist, initialize a new dictionary
        if id not in treatment_embeddings:
            treatment_embeddings[id] = []
        if id not in medication_embeddings:
            medication_embeddings[id] = []
        if id not in diagnosis_embeddings:
            diagnosis_embeddings[id] = []

        for j in range(len(types)):
            t = types[j]
            if t == 'treatment':
                treatment_embedding = codes[j]
                treatment_embeddings[id].append(treatment_embedding)
            elif t == 'medication':
                medication_embedding = codes[j]
                medication_embeddings[id].append(medication_embedding)
            else:
                diagnosis_embedding = codes[j]
                diagnosis_embeddings[id].append(diagnosis_embedding)

    # remove empty values
    treatment_embeddings = {key: value for key, value in treatment_embeddings.items() if value}
    medication_embeddings = {key: value for key, value in medication_embeddings.items() if value}
    diagnosis_embeddings = {key: value for key, value in diagnosis_embeddings.items() if value}

    save_tsv(treatment_embeddings, "./src/processed_data/embeddings/", "treatment_embeddings.tsv")
    save_tsv(medication_embeddings, "./src/processed_data/embeddings/", "medication_embeddings.tsv")
    save_tsv(diagnosis_embeddings, "./src/processed_data/embeddings/", "diagnosis_embeddings.tsv")
