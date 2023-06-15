import csv
import pickle

import numpy as np

def save_text(text: str, save_dir):
    with open(save_dir, 'w') as f:
        f.write(text)

def save_list_to_tsv(data_list, save_dir, delimiter='\t'):
    with open(save_dir, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        for row_data in data_list:
            if not isinstance(row_data, (list, tuple)):
                row_data = [row_data]
            writer.writerow(row_data)


def load_list_from_tsv(save_dir, delimiter='\t', skip_header=False):
    with open(save_dir, 'r') as fin:
        tsv_reader = csv.reader(fin, delimiter=delimiter)
        if skip_header:
            next(tsv_reader)
        return list(tsv_reader)

def save_to_pickle(obj, file_dir: str):
    with open(file_dir, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(file_dir):
    with open(file_dir, 'rb') as f:
        obj = pickle.load(f)
    return obj


def shuffle_list(data_list, seed=5050):
    rdm = np.random.RandomState(seed)
    rnd_indices = rdm.permutation(len(data_list))
    shuffled_data = [data_list[i] for i in rnd_indices]

    return shuffled_data
