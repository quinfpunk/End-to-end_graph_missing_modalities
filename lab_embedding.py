import numpy as np
import torch
from embedding.bert_embedding import *
from tqdm import tqdm
from src.dataset.eicu_dataset import eICUDataset
import json

def build_lab_vocab(train_list, save_dict=True, output_file="lab_vocab.json"):
    """
        @brief:
            Build the vocabulary for the given list of labs
        @params:
            train_list: Lab data in its raw form, for multiple patients
            save_dict: boolean to indicate if the vocabulary must be saved in a file or not
            output_file: name of the output file (advice .json at the end the file gonna be a json anyway; will be used only if save_dict is True !
        @returns:
            The vocab dictionary 
    """
    test_names_set = set()
    for elt in train_list:
        if len(elt) == 0:
            continue
        for patient_value in elt:
            _, tests = patient_value
            for test_name, _ in tests:
                test_names_set.add(test_name)
    test_names_list = sorted(list(test_names_set))
    name2idx = {name: idx for idx, name in enumerate(test_names_list)}
    vocab_size = len(name2idx)
    if save_dict:
        with open(output_file, 'w') as fp:
            json.dump(name2idx, fp, indent=4)
            print('vocab saved successfully to file')
    return name2idx

def embed_lab(data, window_size, lab_vocab, path_to_model="embedding/lab_model.pth", save=False, output_file="saved_lab_embedding"):
    """
        @brief: embed the lab data of a patient
        @params:
            data: The lab data of a patient
            lab_vocab: The vocab of the lab names in our dataset
            path_to_model: model to the pretrained embedding model
            save: boolean to indicate if the emebdding must be saved to a file
            output_file: name of the file where the embedding should be saved; will be used only if save is True !
    """

    # [ 
    # segment the data into windows
    windows = []
    window = segment_time_series(d, window_size)
    windows.append(window)
    print("Segmented Windows:")
    for i, window in enumerate(windows):
        print(f"Window {i}: {window}")
    
    # build a vocabulary for test names
    # we should precompute this from your full dataset
    name2idx = lab_vocab

    names = []
    values = []
    window_transformed = []
    # only on element is expected in data
    window = windows[0][0]
    for _, tests in window:
        for test_name, test_value in tests:
            names.append(name2idx[test_name])
            values.append(test_value)
    test_names_tensor = torch.tensor(names, dtype=torch.long)
    test_values_tensor = torch.tensor(values, dtype=torch.float)
    window_transformed.append((test_names_tensor, test_values_tensor))
    
    # define model hyperparameters
    vocab_size = len(lab_vocab)
    name_embed_dim = 8
    value_dim = 4
    hidden_dim = 16
    num_layers = 2
    num_heads = 2
    # TODO: find a make sense value, biggest in the whole dataset
    # here it is 330
    max_seq_len = 329 + 1  # ensure this is large enough for your sequences
    
    # instantiate the model
    model = TimeSeriesBERTCLS(vocab_size, name_embed_dim, value_dim, hidden_dim, 
                           num_layers, num_heads, max_seq_len)
    model.load_state_dict(torch.load(path_to_model, weights_only=True))
    model.eval()
    
    window_embeddings = []
    for test_names, test_values in window_transformed:
        emb = model.window_encoder(test_names, test_values)
        window_embeddings.append(emb)
    if len(window_embeddings) == 0:
        return []
    # optionally pad/truncate window_embeddings to self.max_seq_len
    # TODO: pad with average
    seq_len = len(window_embeddings)
    if seq_len < model.max_seq_len:
        pad_tensor = torch.zeros(model.max_seq_len - seq_len, window_embeddings[0].shape[-1],
                                 device=window_embeddings[0].device)
        window_embeddings = window_embeddings + [pad_tensor[i] for i in range(pad_tensor.size(0))]
    else:
        window_embeddings = window_embeddings[:model.max_seq_len]
    
    # stack and prepend the CLS token: shape becomes (max_seq_len+1, hidden_dim)
    window_embeddings = torch.stack(window_embeddings, dim=0)
    # perform a forward pass.
    output = model(window_embeddings)  # expected shape: (seq_len, hidden_dim)
    if save:
        f = open("saved_lab_embedding", "a")
        # print("\nTransformer Output:")
        print(output, file=f)
    return output

if __name__ == "__main__":
    # each entry is a tuple: (timestamp, [(test_name, test_value), ...])
    train_set = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")

    data = [] 
    for i in tqdm(range(len(train_set))):
        patient_data = train_set[i]
        patient_lab = patient_data["lab"]
        data.append(patient_lab)
    window_size = 60  # group data points that occur within the same 
    vocab = build_lab_vocab(data, save_dict=False)
    d = [(-117, [('-basos', -0.49131813729984514), ('-eos', -0.33160863632782434), ('-lymphs', 0.003263844576839029), ('-monos', 0.26289545719770113), ('-polys', -0.34092836008115396), ('Hct', 0.8585903186818046), ('Hgb', 0.8408681794985166), ('MCH', 0.5237478387751789), ('MCHC', 0.38965764257547475), ('MCV', 0.3899881151261971), ('MPV', 0.5365022143287047), ('PT', -0.5719228701715722), ('PT - INR', -0.7546218135092618), ('RBC', 0.5338624497080614), ('RDW', -1.0186317248142958), ('WBC x 1000', -0.21880904535829823), ('platelets x 1000', 0.06822139497944821)]), (368, [('BUN', -0.27723792668838293), ('Hct', 0.402484630236823), ('Hgb', 0.48402043060381067), ('MCH', 0.6795892900274573), ('MCHC', 0.45795206289437007), ('MCV', 0.5368434075964227), ('MPV', 0.608621821204078), ('RBC', 0.13845534945796606), ('RDW', -1.0563735656091733), ('WBC x 1000', -0.44210988289374564), ('anion gap', 0.32626205795766744), ('bicarbonate', -0.4409736430775209), ('calcium', 0.7269884449560474), ('chloride', 0.6074246291409929), ('creatinine', -0.521205180811127), ('glucose', -0.5696007585058107), ('platelets x 1000', -0.18819671282349515), ('potassium', 0.2518498188024184), ('sodium', 0.7543928719384024)])]
    print(embed_lab(d, 60, vocab))
