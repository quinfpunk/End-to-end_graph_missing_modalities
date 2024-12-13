from src.eicu_dataset import eICUDataset
import torch

import os
import csv


def save_tsv(list, path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file=file_path, mode="w", newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(list)
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
    treatment_embeddings = [['id', 'embedding']]
    medication_embeddings = [['id', 'embedding']]
    diagnosis_embeddings = [['id', 'embedding']]
    for i in range(train_set.__len__()):
        dict_i = train_set.__getitem__(i)
        types = dict_i["types"]
        id = dict_i["id"]
        codes = dict_i["codes"][1:-1]
        for j in range(len(types)):
            t = types[j]
            if t == 'treatment':
                treatment_embedding = codes[j]
                d = [id, treatment_embedding]
                treatment_embeddings.append(d)
            elif t == 'medication':
                medication_embedding = codes[j]
                d = [id, medication_embedding]
                medication_embeddings.append(d)
            else:
                diagnosis_embedding = codes[j]
                d = [id, diagnosis_embedding]
                diagnosis_embeddings.append(d)

    save_tsv(treatment_embeddings, "./src/processed_data/embeddings/", "treatment_embeddings.tsv")
    save_tsv(medication_embeddings, "./src/processed_data/embeddings/", "medication_embeddings.tsv")
    save_tsv(diagnosis_embeddings, "./src/processed_data/embeddings/", "diagnosis_embeddings.tsv")
