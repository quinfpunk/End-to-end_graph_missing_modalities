import torch.nn as nn 
from sentence_transformers import SentenceTransformer
from embedding.bert_embedding import * 
from lab_embedding import *
from src.dataset.eicu_dataset import eICUDataset


def preprocess_row(row):
    # row is a list of tuples, e.g., [("test_name", "test_value"), ("test_name2", "test_value2")]     
    # Format each tuple as "name: value" and join them with a delimiter (space or comma)    
    return " ".join([f"{name}: {value}" for name, value in row]) 

def embed_lab(data, window_size, model_name="all-MiniLM-L6-v2", save=False, output_file="saved_lab_embedding"):

    model = SentenceTransformer(model_name) 
    window = segment_time_series(data, window_size)
    # print("Segmented Windows:")
    # print(f"Window {i}: {window}")
    
    # only on element is expected in data
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


if __name__ == "__main__":
    
    train_set = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")

    data = [] 
    for i in tqdm(range(len(train_set))):
        patient_data = train_set[i]
        patient_lab = patient_data["lab"]
        data.append(patient_lab)
    window_size = 60
    all_embeddings = []
    for d in tqdm(data):
        all_embeddings.append(embed_lab(d, 60))
    all_embeddings = np.array(all_embeddings)
    with open('small_eicu_lab_embeddings.npy', 'wb') as f:   
        np.save(f, all_embeddings) 


