import os
os.environ['http_proxy'] = "http://127.0.0.1:27890"
os.environ['https_proxy'] = "http://127.0.0.1:27890"

import argparse
from my_utils.my_dataloader import load_test_input
from my_utils.data_utils import save_list_to_tsv, load_from_pickle
from sico.prompt_constructor import PromptConstructor
from sico.LLM_api import get_llm_api
from pathlib import Path
from my_utils.text_utils import replace_changeline
from detectors import RoBERTaAIDetector
from tqdm import tqdm
import shared_dir
import numpy as np


def load_incontext_data(save_dir):
    print(f'Load from {save_dir}')
    loaded = load_from_pickle(save_dir)

    return loaded

def SICO_gen(final_prompt, test_data_list, llm_api):


    # construct the input to LLM: t_feature + p_task + (x_ic, y_ic) + x_eval
    llm_input_list = [final_prompt.format(test_data) for test_data in test_data_list]

    print('Start generating on test set')
    # generate from LLM
    test_task_outputs = []
    in_out_pairs = []
    for i, llm_input in enumerate(tqdm(llm_input_list)):
        cur_n_token = None
        test_task_output = llm_api(llm_input, cur_n_token, 1, {'temperature': 1})[0]

        test_task_output = replace_changeline(test_task_output)
        test_task_outputs.append(test_task_output)

        in_out_pairs.append((test_data_list[i], test_task_output))

    in_out_pairs.insert(0, ('input', 'SICO-output'))

    return in_out_pairs, test_task_outputs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='squad',
                        choices=['squad', 'yelp', 'eli5'])
    parser.add_argument('--detector', type=str, default='chatdetect',
                        choices=['chatdetect', 'gpt2detect', 'gptzero', 'openai', 'detectgpt', 'logrank'])
    parser.add_argument('--llm', type=str, default='chatgpt', choices=['chatgpt', 'vicuna'])
    parser.add_argument('--task', type=str, default='essay', choices=['essay', 'qa', 'rev-gen', 'paraphrase'])

    parser.add_argument('--incontext-size', type=int, default=8)
    parser.add_argument('--eval-size', type=int, default=32)
    parser.add_argument('--train-iter', type=int, default=6)
    parser.add_argument('--para-num', type=int, default=8)

    parser.add_argument('--test-size', type=int, default=200)

    args = parser.parse_args()

    detector_name = args.detector
    dataset = args.dataset
    llm_name = args.llm
    task_type = args.task

    eval_size = args.eval_size
    max_train_iter = args.train_iter
    incontext_size = args.incontext_size
    para_num = args.para_num


    tag = f'run_{args.dataset}_{task_type}_{llm_name}_{detector_name}_eval={eval_size}_ic={incontext_size}_iter={max_train_iter}'
    save_dir = Path(shared_dir.train_result_dir, tag, 'best_feature_ice.pkl')

    # load dir
    test_data_list = load_test_input(dataset_name=dataset, task_name=task_type)[:args.test_size]

    feature, incontext_examples = load_incontext_data(save_dir)
    prompt_constructor = PromptConstructor(task_type=args.task)
    final_prompt = prompt_constructor.get_final_prompt(feature, incontext_examples)
    llm_api = get_llm_api(args.llm, input_device=0)

    all_data_list, generated_outputs = SICO_gen(final_prompt, test_data_list, llm_api)


    # save
    save_name = f'SICO-{args.dataset}-{args.task}-{args.llm}-{args.detector}'
    output_folder = Path(shared_dir.test_results_dir, args.dataset, save_name)
    output_folder.mkdir(parents=True, exist_ok=True)
    save_list_to_tsv(all_data_list, output_folder.joinpath('generated_text.tsv'))


    # # use chatdetect to evaluate
    #
    # detector = RoBERTaAIDetector(0)
    #
    # ai_probs, preds = detector(generated_outputs)
    #
    # acc = (preds == np.ones(len(generated_outputs), dtype=int)).sum() / len(generated_outputs)
    # print('ChatGPT detector')
    # print('Average AI scores:', np.mean(ai_probs))
    # print(f'Detection Accuracy: {acc:.2%}')











