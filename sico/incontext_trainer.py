import logging
from pathlib import Path
import copy
import pickle

import torch
import functools
from tqdm import tqdm
import numpy as np

import wandb

import torch
import datetime
from pathlib import Path

from transformers import set_seed

import shared_dir

from my_utils.my_logger import MyLogger

from detectors import RoBERTaAIDetector, OpenAIDetector, RankDetector, DetectGPT, GPTZeroDetector

from my_utils.data_utils import save_list_to_tsv, save_to_pickle, load_from_pickle
from sico.cand_generator import WordNetCandGenerator, ParaLLMGenerator
from sico.LLM_api import get_llm_api
from sico.prompt_constructor import PromptConstructor
from my_utils.text_utils import replace_changeline
from sico.context_optimizer import context_text_optimization
from my_utils.my_dataloader import load_eval_data


class SICOTrainer:

    def __init__(self, dataset, llm_name, detector_name, task_type, eval_size=32, ic_num=8, max_train_iter=6,
                 gen_kwargs=(),
                 para_num=8, save_dir='./', tag='', seed=5050, disable_wandb=True
                 ):

        set_seed(seed)
        self.time_stamp = datetime.datetime.now().strftime('%m%d_%H%M%S')

        # logger
        log_filename = 'train_{}_{}_{}_{}_{}.log'.format(dataset, task_type, llm_name, detector_name, self.time_stamp)
        log_dir = shared_dir.log_folder_dir + log_filename
        self.logger = MyLogger(log_dir, level=logging.DEBUG)  # DEBUG, INFO

        self.eval_size = eval_size

        self.incontext_example_num = ic_num
        self.max_train_iter = max_train_iter

        self.max_edit_iter = 1
        self.max_word_change_rate = 1
        self.max_sent_change_rate = 1
        self.cand_type = 'Sub-Para/Word'
        self.gen_kwargs = gen_kwargs
        self.init_feature_num = 5

        self.dataset = dataset  # squad, eli5
        self.llm_name = llm_name  # chatgpt, gpt2
        self.detector_name = detector_name
        self.task_type = task_type

        train_config = {
            'Dataset': self.dataset,
            'LLM': self.llm_name,
            'Detector for training': self.detector_name,
            'Task type': self.task_type,
            'Generation Args:': self.gen_kwargs,
            'Eval size': self.eval_size,
            'ICE number': self.incontext_example_num,
            'Edit': f'{self.cand_type}:[Word:{self.max_word_change_rate}, Sent-{para_num}:{self.max_sent_change_rate}]',
            'seed': seed
        }

        params_info = '\nHyper-params\n'
        for k in train_config:
            params_info += f'{k}={train_config[k]}\n'
        self.logger.info(params_info)
        print(params_info)

        # init wandb
        proj_name = f'{dataset}_{llm_name}_{detector_name}_{task_type}'
        run_name = f'Opt={self.cand_type},EvalSize={self.eval_size},#ICE={self.incontext_example_num},T={self.time_stamp}'
        mode = 'disabled' if disable_wandb else None
        wandb.init(project=proj_name, config=train_config,
                   name=run_name, mode=mode)


        # init base models
        # all gpus
        available_gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.input_device = available_gpus[0]
        self.detector_device = available_gpus[0]
        generator_device = available_gpus[1]

        # proxy detector

        if detector_name == 'chatdetect':
            detector = RoBERTaAIDetector(self.detector_device)
        elif detector_name == 'openai':
            detector = OpenAIDetector(shared_dir.cache_folder_dir + f'train_{dataset}_{detector_name}.cache')
        elif detector_name == 'logrank':
            detector = RankDetector(self.detector_device, log_rank=True)
        elif detector_name == 'detectgpt':
            detector = DetectGPT(threshold=0.5, sample_num=100, mask_device=self.detector_device,
                                 base_device=self.detector_device,
                                 cache_path=shared_dir.cache_folder_dir + f'train_{dataset}_{detector_name}.cache',
                                 use_cache=True)
        elif detector_name == 'gptzero':
            detector = GPTZeroDetector(
                cache_path=shared_dir.cache_folder_dir + f'train_{dataset}_{detector_name}.cache')
        else:
            raise Exception

        self.detector = detector

        # LLM
        self.llm_api = get_llm_api(self.llm_name, self.input_device)

        # candidate generator for optimization
        if task_type == 'paraphrase':
            threshold_ = 1e-4
        else:
            threshold_ = 1e-5
        self.word_cand_generator = WordNetCandGenerator(mlm_conf_threshold=threshold_, device=generator_device)
        self.para_cand_generator = ParaLLMGenerator(self.llm_api, gen_kwargs, para_num=para_num)

        # prompt constructor
        self.prompt_constructor = PromptConstructor(self.task_type)

        # load dataset
        # train_data_list, test_data_list = load_train_test_data(dataset, self.detector, self.K_shot)
        self.eval_data_list = load_eval_data(dataset_name=self.dataset, task_name=self.task_type,
                                             eval_size=self.eval_size)

        # training record
        self.best_incontext_examples = None
        self.feature = None
        self.best_prompt_text = ''
        self.best_score = -9999
        self.best_acc = 100

        self.best_dev_acc = 100
        self.init_sum_num = 5

        # # hyper para
        self.total_iter = max_train_iter
        # self.max_edit_iter = max_edit_iter
        # self.max_change_rate = max_change_rate

        self.ai_label = 1
        self.human_label = 0

        # save results
        self.tag = tag
        if save_dir:
            self.save_path = Path(save_dir, tag)
            self.save_path.mkdir(parents=True, exist_ok=True)

        self.icd_pickle_filename = '{}_feature_ice.pkl'
        self.prompt_text_filename = '{}_final_prompt.txt'

        self.query_num_list = []
        self.prompt_history_list = []

    def extract_feature(self, human_task_outputs, ai_task_outputs):
        # Step 1: Ask LLM summarize and extract the difference of AI and human
        print('Step 1: Construct feature t_feature')

        # construct prompt for feature extraction
        extract_feature_prompt = self.prompt_constructor.prompt_extract_feature(human_task_outputs=human_task_outputs,
                                                                                ai_task_outputs=ai_task_outputs)

        cur_n_token = len(extract_feature_prompt.split()) * 1.2
        feature_list = self.llm_api(extract_feature_prompt, cur_n_token, self.init_feature_num, {'temperature': 0.9})

        # evaluate the features and select the best
        utility_score_list = []
        final_prompt_list = []
        for cur_feature in feature_list:
            cur_final_prompt = self.prompt_constructor.get_final_prompt(cur_feature, [])
            cur_utility_score, _acc = self.evaluate_prompt(cur_final_prompt)

            utility_score_list.append(cur_utility_score)
            final_prompt_list.append(cur_final_prompt)

        best_idx = np.argmax(utility_score_list)

        best_final_prompt = final_prompt_list[best_idx]
        best_utility_score = utility_score_list[best_idx]
        best_feature = feature_list[best_idx]

        msg_ = '\n ================= Init Feature ======================\n' \
               f'Best U-Score: {best_utility_score}\n' \
               f'Best Prompt: \n{best_final_prompt}\n' \
                f'Best Feature:\n{best_feature}\n' \
                '======================================================'

        self.logger.info(msg_)
        print(msg_)

        return best_feature

    def construct_incontext_outputs(self, feature_text, ai_task_outputs):
        # Step 2: build in-context outputs y_ic
        # using extracted feature t_feature to paraphrase AI outputs y_AI
        print('Step 2: Construct y_ic')

        paraphrase_final_prompt = self.prompt_constructor.get_final_prompt_paraphrase(feature_text, [])

        # get text from api
        llm_input_list = [paraphrase_final_prompt.format(ai_output) for ai_output in ai_task_outputs]

        task_outputs_ic = []
        for i, llm_input in enumerate(tqdm(llm_input_list)):
            cur_n_token = len(ai_task_outputs) // 2
            generated_t = self.llm_api(llm_input, cur_n_token, 1, self.gen_kwargs)[0]

            generated_t = replace_changeline(generated_t)
            task_outputs_ic.append(generated_t)

            self.logger.debug(f'{llm_input} || {generated_t}')

        return task_outputs_ic

    def evaluate_prompt(self, final_prompt, return_text=False, disable_tqdm=False):
        # calculate utility score in Equation(1)

        # construct the input to LLM: t_feature + p_task + (x_ic, y_ic) + x_eval
        llm_input_list = [final_prompt.format(task_input_eval) for task_input_eval in self.eval_data_list]

        # generate from LLM
        eval_task_outputs = []
        for i, llm_input in enumerate(tqdm(llm_input_list, disable=disable_tqdm)):
            cur_n_token = None
            eval_task_output = self.llm_api(llm_input, cur_n_token, 1, self.gen_kwargs)[0]

            eval_task_output = replace_changeline(eval_task_output)
            eval_task_outputs.append(eval_task_output)

            self.logger.debug(f'{llm_input} || {eval_task_output}')

        # calculate utility score: 1 - AVG(AI-probs)
        ai_score_list, label_list = self.detector(eval_task_outputs)

        gt_label = np.ones(len(eval_task_outputs), dtype=int)
        acc = (np.array(label_list) == gt_label).sum() / len(eval_task_outputs)
        U_score = 1 - np.mean(ai_score_list)

        if return_text:
            return U_score, acc, eval_task_outputs
        else:
            return U_score, acc

    def train(self, init_data_list):

        task_inputs, task_outputs_human, task_outputs_ai = list(zip(*init_data_list))  # x_ic, y_human, y_ai

        # init prompt from chatgpt
        # step 1: get feature
        self.feature = self.extract_feature(task_outputs_human, task_outputs_ai)

        # step 2: construct in-context outputs y_ic by paraphrasing
        task_outputs_ic = self.construct_incontext_outputs(self.feature, task_outputs_ai)

        # init in-context examples
        self.best_incontext_examples = copy.deepcopy(list(zip(task_inputs, task_outputs_ic)))
        self.best_prompt_text = self.prompt_constructor.get_final_prompt(self.feature, self.best_incontext_examples)

        # evaluate
        self.best_score, self.best_acc = self.evaluate_prompt(self.best_prompt_text)

        self.save_incontext_data((self.feature, self.best_incontext_examples), self.best_prompt_text, tag_=0)

        self.logger.info('================ Start training =======================')
        self.logger.info(f'Init Context Loss {self.best_score}, Acc: {self.best_acc:.2%}')
        wandb.log({'Utility Score': self.best_score, 'Accuracy': self.best_acc}, step=0)
        wandb.log({'Best Utility Score': self.best_score, 'Best Accuracy': self.best_acc}, step=0)

        # step 3: Substitution-based in-context optimization
        train_i = 1
        with tqdm(initial=train_i, total=self.total_iter) as pbar:
            while True:

                # (1) sentence substitution
                new_ic_examples, new_ic_score = self._optimize_ic_outputs(self.best_incontext_examples, 'sent')

                wandb.log({'ICD Human Score': new_ic_score}, step=train_i)

                # eval new in-context examples
                sent_update = self.eval_and_save(new_ic_examples, train_i)

                train_i += 1
                pbar.update(1)
                if train_i > self.total_iter:
                    break

                # (2) word substitution
                new_ic_examples, new_ic_score = self._optimize_ic_outputs(self.best_incontext_examples, 'word')
                wandb.log({'ICD Human Score': new_ic_score}, step=train_i)

                # eval new in-context examples
                word_update = self.eval_and_save(new_ic_examples, train_i)

                train_i += 1
                pbar.update(1)
                if train_i > self.total_iter:
                    break

    def eval_and_save(self, new_ic_examples, step):

        cur_final_prompt = self.prompt_constructor.get_final_prompt(self.feature, new_ic_examples)

        new_score, new_acc, new_texts = self.evaluate_prompt(cur_final_prompt, return_text=True, disable_tqdm=True)
        self.save_data_list(list(zip(self.eval_data_list, new_texts)), tag=step)

        wandb.log({'Utility Score': new_score, 'Accuracy': new_acc}, step=step)

        is_update = False

        # save iter
        self.save_incontext_data((self.feature, new_ic_examples), cur_final_prompt, tag_=step)
        self.logger.info(f'Iter {step} Prompt:\n{cur_final_prompt}')

        if new_score > self.best_score:
            self.best_score = new_score
            self.best_acc = new_acc
            # self.best_icd_list = new_icd_list
            self.logger.info(f'-- Iter {step} find better score: {self.best_score: .4f}, acc: {self.best_acc:.2%}')

            self.best_incontext_examples = new_ic_examples
            self.best_prompt_text = cur_final_prompt
            self.save_incontext_data((self.feature, self.best_incontext_examples), self.best_prompt_text)
            is_update = True

        wandb.log({'Best Utility Score': self.best_score, 'Best Accuracy': self.best_acc}, step=step)

        return is_update

    def save_incontext_data(self, feature_incontext_examples, final_prompt, tag_='best'):
        self.logger.info(f'Save to {self.save_path}')
        if feature_incontext_examples:
            save_to_pickle(feature_incontext_examples, self.save_path.joinpath(self.icd_pickle_filename.format(tag_)))

        if final_prompt:
            with open(self.save_path.joinpath(self.prompt_text_filename.format(tag_)), 'w') as f:
                f.write(final_prompt)



    def save_data_list(self, text_data_list, tag='normal'):
        cur_filename = f'text_{tag}.tsv'
        save_dir = self.save_path.joinpath(cur_filename)
        save_list_to_tsv(text_data_list, save_dir)
        print(f'Save to {save_dir}')


    def _optimize_ic_outputs(self, ic_examples, edit_type):

        # optimization goal: maximize human-score = 1 - AI-prob
        def _human_score(text_list):
            ai_score_list, _ = self.detector(text_list)  # N * 2

            human_score_list = [1 - d for d in ai_score_list]
            return human_score_list

        new_ic_examples = []
        human_score_list = []

        for i in range(len(ic_examples)):
            cur_data = ic_examples[i]

            cur_x_ic = cur_data[0]
            cur_y_ic = cur_data[1]

            # generate candidates
            if edit_type == 'sent':
                paraphrase_template = self.prompt_constructor.get_final_prompt_paraphrase(self.feature, [])

                y_part_list, y_idx_part_list, y_cand_dict = self.para_cand_generator.generate_para_dict(
                    cur_y_ic, paraphrase_template)
            elif edit_type == 'word':
                y_part_list, y_idx_part_list, y_cand_dict = self.word_cand_generator.generate_cand_dict(
                    cur_y_ic)
            else:
                raise Exception('Wrong edit type', edit_type)

            new_y_ic, new_human_score, query_num, edit_num = context_text_optimization(
                start_text=cur_y_ic,
                start_part_list=y_part_list,
                idx_part_list=y_idx_part_list,
                cand_dict=y_cand_dict,
                eval_f=_human_score,
                max_iter=self.max_edit_iter, change_rate=1)

            new_ic_examples.append((cur_x_ic, new_y_ic))
            self.query_num_list.append(query_num)
            human_score_list.append(new_human_score)

        return new_ic_examples, np.mean(human_score_list)
