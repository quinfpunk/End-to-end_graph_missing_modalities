import torch.nn as nn 
from sentence_transformers import SentenceTransformer
from embedding.bert_embedding import * 
from lab_embedding import *
from src.dataset.eicu_dataset import eICUDataset
import argparse
import os.path 
from multiprocessing import Pool 
import multiprocessing
import pickle
import os
from src.utils import processed_data_path

def merge_pickles(prefix_name: str, path: str):
    """
        @brief:
            Merge pickles at path containing prefix name
    """
    res = {}
    for file in os.listdir(path):
        if prefix_name in f and ".pkl" in file:
            # open file and load pickle update the dictionary
            with open(file, "rb") as f:
                tmp = pickle.load(f)
                res.update(tmp)
    return res

     


def preprocess_row(row):
    # row is a list of tuples, e.g., [("test_name", "test_value"), ("test_name2", "test_value2")]     
    # Format each tuple as "name: value" and join them with a delimiter (space or comma)    
    return " ".join([f"{name}: {value}" for name, value in row]) 

def embed_lab(data, window_size, model_name="all-MiniLM-L6-v2", save=False, output_file="saved_lab_embedding"):
    """
        @brief:
            embed data using the model given in parameters.
            The results of this specific embedding can be save in output_file.
    """

    # device cpu to allow multiprocessing with GPU multiprocessing creates errors
    model = SentenceTransformer(model_name, device="cpu") 
    window = segment_time_series(data, window_size)
    
    # stringify data then embed it
    embeddings = []
    global_sentence = ""
    for tmp in window:
        global_sentence += "{tmp}: ("
        for _, tests in tmp:
            sentence = preprocess_row(tests)  
            # convert the row to a single string     
            global_sentence += sentence
        global_sentence += "), "
    embedding = model.encode(global_sentence)     
    embeddings.append(embedding)
    if save:
        f = open(output_file, "a")
        print(output, file=f)
    return embeddings

def save_embedding(dataset, filename=os.path.join(processed_data_path, "lab_embedding")):
    """
        @brief:
            Use multiprocessing to speed up the embedding process
            Saving the embeddings in multiple pickles for better multiprocessing 
            Saved file will be saved at filename path with _<number> as a suffix
        @returns:
            Returns a dictionary containing all embedding
    """
    data = {} 
    for icu_id, icu_stay in tqdm(dataset.items()):
        patient_lab = icu_stay.lab
        data[icu_id] = patient_lab
    window_size = 60
    # init pool of process
    pool = Pool()
    # number of logical thread is extracted 
    max_processes = multiprocessing.cpu_count()
    
    lab_data = list(data.values())
    patient_ids = list(data.keys())
    for shards in tqdm(range(int(len(data) / max_processes))):
        all_embeddings = {}
        for idx, d in tqdm(enumerate(lab_data[shards * max_processes: (shards + 1) * max_processes])):
            embed = pool.apply_async(embed_lab, [d, 60])
            all_embeddings[patient_ids[idx]] = embed # (embed_lab(d, 60))

        for idx  in range(len(all_embeddings.values())):
            all_embeddings[list(all_embeddings.keys())[idx]] = list(all_embeddings.values())[idx].get()
            
        if os.path.isfile(f'{filename}_{shards}.pkl'):
            print(f'{filename}_{shards}.pkl already exists. It will be overwrite !')
        with open(f'{filename}_{shards}.pkl', 'wb') as f:   
            pickle.dump(all_embeddings, f)
    rest_to_compute = int(len(data) / max_processes) * max_processes
    pool.close()
    # No parallelism anymore as we have less than max_processe embedding to do
    if  (len(data) % max_processes) != 0:
        all_embeddings = {}
        for idx, d in tqdm(enumerate(lab_data[rest_to_compute:])):
            all_embeddings[patient_ids[idx]] = embed_lab(d, 60)

        if os.path.isfile(f'{filename}_{shards}.pkl'):
            print(f'{filename}_{shards}.pkl already exists. It will be overwrite !')
        with open(f'{filename}_{(rest_to_compute / max_processes) + 1}.pkl', 'wb') as f:   
            pickle.dump(all_embeddings, f)
    print("done !")

    head, tail = os.path.split(filename)
    # use merge pickle to return a dictionary
    return merge_pickles(tail, head)





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(               
                                     prog='Embedding',                    
                                     description='Generate lab embedding. Pay attention to have the icu_stay_dict.pkl generated before running the script ! ',                     
                                     epilog='A simple use of the script would be "python sentence_transformer_embedding.py". This would embed the whole train dataset and store the results into the processed_data_path.')
    parser.add_argument('-f', '--filename', default=os.path.join(processed_data_path, "lab_embedding"))
    parser.add_argument('--dataset_size', default="")
    parser.add_argument('--split', default="train")
    args = parser.parse_args()
    train_set = eICUDataset(split=args.split, task="mortality", load_no_label=True, data_size=args.dataset_size)

    # This code is the same as save_embedding
    # It allows the embedding saving to be done as a script 
    data = {} 
    for i in tqdm(range(len(train_set))):
        patient_data = train_set[i]
        patient_lab = patient_data["lab"]
        data[patient_data["id"]] = patient_lab
    window_size = 60
    pool = Pool()
    max_processes = multiprocessing.cpu_count()
    
    lab_data = list(data.values())
    patient_ids = list(data.keys())
    for shards in tqdm(range(int(len(data) / max_processes))):
        all_embeddings = {}
        for idx, d in tqdm(enumerate(lab_data[shards * max_processes: (shards + 1) * max_processes])):
            embed = pool.apply_async(embed_lab, [d, 60])
            all_embeddings[patient_ids[idx]] = embed # (embed_lab(d, 60))

        for idx  in range(len(all_embeddings.values())):
            all_embeddings[list(all_embeddings.keys())[idx]] = list(all_embeddings.values())[idx].get()
            
        if os.path.isfile(f'{args.filename}_{shards}.pkl'):
            print(f'{args.filename}_{shards}.pkl already exists. It will be overwrite !')
        with open(f'{args.filename}_{shards}.pkl', 'wb') as f:   
            pickle.dump(all_embeddings, f)
    rest_to_compute = int(len(data) / max_processes) * max_processes
    pool.close()
    if  (len(data) % max_processes) != 0:
        all_embeddings = {}
        for idx, d in tqdm(enumerate(lab_data[rest_to_compute:])):
            all_embeddings[patient_ids[idx]] = embed_lab(d, 60)

        if os.path.isfile(f'{args.filename}_{shards}.pkl'):
            print(f'{args.filename}_{shards}.pkl already exists. It will be overwrite !')
        with open(f'{args.filename}_{(rest_to_compute / max_processes) + 1}.pkl', 'wb') as f:   
            pickle.dump(all_embeddings, f)
    print("done !")



