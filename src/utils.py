import csv
import json
import logging
import os
import pickle
import random
import sys

import numpy as np
import torch

import platform
project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# project_name = os.path.basename(project_path)
project_name = ''
# server_name = os.uname()[1]
server_name = platform.uname()[1]
remote_root = "/Users/daniele/PycharmProjects/End-to-end_graph_missing_modalities" # change

try:
    raw_data_path = os.path.join(remote_root, "raw_data")
    remote_project_path = os.path.join(remote_root, project_name)
    processed_data_path = os.path.join(remote_project_path, "processed_data")
    output_path = os.path.join(remote_project_path, "output")
except:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_csv(filename):
    logging.info(f"Reading from {filename}")
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.DictReader(file, delimiter=",")
        for row in csv_reader:
            data.append(row)
    header = list(data[0].keys())
    return header, data


def read_txt(filename):
    logging.info(f"Reading from {filename}")
    data = []
    with open(filename, "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            data.append(line)
    return data


def write_txt(filename, data):
    logging.info(f"Writing to {filename}")
    with open(filename, "w") as file:
        for line in data:
            file.write(line + "\n")
    return


def read_json(filename):
    logging.info(f"Reading from {filename}")
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def write_json(filename, data):
    logging.info(f"Writing to {filename}")
    with open(filename, "w") as file:
        json.dump(data, file)
    return


def create_directory(directory):
    if not os.path.exists(directory):
        logging.info(f"Creating directory {directory}")
        os.makedirs(directory)


def load_pickle(filename):
    logging.info(f"Data loaded from {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


def dump_pickle(data, filename):
    logging.info(f"Data saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    set_seed(0)
    print(project_path)
    print(project_name)
    print(remote_root)
    print(remote_project_path)
    print(raw_data_path)
    print(processed_data_path)
    print(torch.cuda.is_available())
