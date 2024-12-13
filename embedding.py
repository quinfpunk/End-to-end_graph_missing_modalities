from torch.nn.functional import embedding, Tensor
from torch.onnx.symbolic_opset9 import tensor
from transformers.utils.fx import torch_abs

from src.eicu_dataset import eICUDataset
import torch

import csv
import os


def save_tsv(dict_list, path, file_name):
    headers = dict_list[0].keys()
    file_path = os.path.join(path, file_name)
    with open(file=file_path, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        writer.writerows(dict_list)
    print("Saved ", file_name)

# Press the green button in the gutter to run the script.
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

    labvectors_embeddings = []
    apacheapsvar_embeddings = []
    for i in range(train_set.__len__()):
        dict_i = train_set.__getitem__(i)
    #     if dict_i["gender"].numel() == 0 \
    #         or dict_i["age"].numel() == 0 \
    #         or dict_i["ethnicity"].numel() == 0:
    #         print("Problem")
    # print("Yes")

        if dict_i["labvectors_flag"]:
            id = dict_i["id"]
            labvectors_embedding = dict_i["labvectors"]
            # print(labvectors_embedding.size())
            d = {"id": id, "embedding": labvectors_embedding}
            labvectors_embeddings.append(d)
        if dict_i["apacheapsvar_flag"]:
            id = dict_i["id"]
            apacheapsvar_embedding = dict_i["apacheapsvar"]
            d = {"id": id, "embedding": apacheapsvar_embedding}
            apacheapsvar_embeddings.append(d)

    save_tsv(labvectors_embeddings, "./src/processed_data/embeddings/", "labvectors_embeddings.tsv")
    save_tsv(apacheapsvar_embeddings, "./src/processed_data/embeddings/", "apacheapsvar_embeddings.tsv")
