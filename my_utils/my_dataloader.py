import shared_dir
from .data_utils import load_list_from_tsv


def load_eval_data(dataset_name, task_name, eval_size=32):
    raw_data_list = load_list_from_tsv(shared_dir.dataset_dir + dataset_name + '/eval.tsv', skip_header=True)

    if task_name == 'paraphrase':
        final_data = [d[2] for d in raw_data_list] # x_ic = y_ai
    else:
        final_data = [d[0] for d in raw_data_list] # x_ic

    if len(final_data) < eval_size:
        print('Large eval size', eval_size, 'total number is', len(final_data))

    return final_data[:eval_size]

def load_init_data_list(dataset_name, task_name, ic_size):
    raw_data_list = load_list_from_tsv(shared_dir.dataset_dir + dataset_name + '/incontext.tsv', skip_header=True)

    if task_name == 'paraphrase':
        final_data = [(d[2], d[1], d[2]) for d in raw_data_list] # x_ic = y_ai
    else:
        final_data = [(d[0], d[1], d[2]) for d in raw_data_list] # x_ic, y_human, y_ai

    if len(final_data) < ic_size:
        print('Large in-context size', ic_size, 'total number is', len(final_data))


    return final_data[:ic_size]


def load_test_input(dataset_name, task_name):
    raw_data_list = load_list_from_tsv(shared_dir.dataset_dir + dataset_name + '/test.tsv', skip_header=True)

    if task_name == 'paraphrase':
        final_data = [d[2] for d in raw_data_list]
    else:
        final_data = [d[0] for d in raw_data_list]
    return final_data

def load_test_output_human(dataset_name):
    raw_data_list = load_list_from_tsv(shared_dir.dataset_dir + dataset_name + '/test.tsv', skip_header=True)
    y_human_list = [d[1] for d in raw_data_list]
    return y_human_list

def load_test_output_ai(dataset_name):
    raw_data_list = load_list_from_tsv(shared_dir.dataset_dir + dataset_name + '/test.tsv', skip_header=True)
    y_ai_list = [d[2] for d in raw_data_list]
    return y_ai_list
