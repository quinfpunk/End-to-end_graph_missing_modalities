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
    print("Segmented Windows:")
    print(f"Window {i}: {window}")
    
    # only on element is expected in data
    embeddings = []
    global_sentence = ""
    for tmp in window:
        print(tmp)
        global_sentence += "{tmp}: ("
        for _, tests in tmp:
            print(tests)
            sentence = preprocess_row(tests)  
            # convert the row to a single string     
            global_sentence += sentence
        global_sentence += "), "
    embedding = model.encode(global_sentence)     
    embeddings.append(embedding)
    if save:
        f = open(output_file, "a")
        print(output, file=f)
    print(len(embeddings[0]))# , len(embeddings[1]))
    return embeddings


if __name__ == "__main__":
    train_set = eICUDataset(split="train", task="mortality", load_no_label=True, data_size="big")

    data = [] 
    for i in tqdm(range(len(train_set))):
        patient_data = train_set[i]
        patient_lab = patient_data["lab"]
        data.append(patient_lab)
    d =  [(-117, [('-basos', -0.49131813729984514), ('-eos', -0.33160863632782434), ('-lymphs', 0.003263844576839029), ('-monos', 0.26289545719770113), ('-polys', -0.34092836008115396), ('Hct', 0.8585903186818046), ('Hgb', 0.8408681794985166), ('MCH', 0.5237478387751789), ('MCHC', 0.38965764257547475), ('MCV', 0.3899881151261971), ('MPV', 0.5365022143287047), ('PT', -0.5719228701715722), ('PT - INR', -0.7546218135092618), ('RBC', 0.5338624497080614), ('RDW', -1.0186317248142958), ('WBC x 1000', -0.21880904535829823), ('platelets x 1000', 0.06822139497944821)]), (368, [('BUN', -0.27723792668838293), ('Hct', 0.402484630236823), ('Hgb', 0.48402043060381067), ('MCH', 0.6795892900274573), ('MCHC', 0.45795206289437007), ('MCV', 0.5368434075964227), ('MPV', 0.608621821204078), ('RBC', 0.13845534945796606), ('RDW', -1.0563735656091733), ('WBC x 1000', -0.44210988289374564), ('anion gap', 0.32626205795766744), ('bicarbonate', -0.4409736430775209), ('calcium', 0.7269884449560474), ('chloride', 0.6074246291409929), ('creatinine', -0.521205180811127), ('glucose', -0.5696007585058107), ('platelets x 1000', -0.18819671282349515), ('potassium', 0.2518498188024184), ('sodium', 0.7543928719384024)])]
    d2 = [(99, [('creatinine', -0.47844800004328386)])]
    window_size = 60
    print(embed_lab(d, 60))
