import os
import time

import re
import torch
import numpy as np
import openai
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast, T5Tokenizer,  RobertaForSequenceClassification, RobertaTokenizer, AutoModelForSeq2SeqLM

from my_utils.data_utils import load_from_pickle, save_to_pickle

from my_utils.model_path import get_model_path










class AIDetector:

    def __call__(self, text_list, disable_tqdm=True):
        raise NotImplementedError

    def get_threshold(self):
        raise NotImplementedError

    def save_cache(self):
        return


class RoBERTaAIDetector(AIDetector):

    def __init__(self, device, name='chatdetect', batch_size=64):

        if name == 'chatdetect':
            model_name = get_model_path("Hello-SimpleAI/chatgpt-detector-roberta")
            self.ai_label = 1
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif name == 'gpt2detect':
            model_name = get_model_path('roberta-base-openai-detector')
            self.ai_label = 0
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name)
            self.tokenizer.model_max_length = 512
        else:
            raise Exception('Wrong name', name)

        self.device = device
        self.model.to(device)
        self.batch_size = batch_size

    def __call__(self, text_list, disable_tqdm=True):
        num_examples = len(text_list)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size

        batch_input_list = []
        for i in range(num_batches):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, num_examples)
            batch_inputs = self.tokenizer(text_list[start_index:end_index], padding=True, truncation=True,
                                          return_tensors="pt")
            batch_inputs = {
                "input_ids": batch_inputs["input_ids"],
                "attention_mask": batch_inputs["attention_mask"],
            }
            batch_input_list.append(batch_inputs)

        logits = []
        labels = []

        with torch.no_grad():
            for batch_inputs in batch_input_list:
                batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                batch_outputs = self.model(**batch_inputs)
                batch_predicted_labels = torch.argmax(batch_outputs.logits, dim=1)
                logits.append(batch_outputs.logits)
                labels.append(batch_predicted_labels)

        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        ai_score_list = (probs[:, self.ai_label]).cpu().numpy().tolist()

        detect_preds = labels.cpu().numpy()
        if self.ai_label == 0: # reverse
            detect_preds = 1 - detect_preds

        # return {'logits': logits, 'labels': labels}
        return ai_score_list, detect_preds

    def get_threshold(self):
        return 0.5



class DetectGPT(AIDetector):

    def __init__(self, threshold=0.9,  sample_num=50, mask_device='cuda:0', base_device='cuda:1', cache_path=None, use_cache=True):

        # init models
        gpt_model_path = get_model_path("gpt2-medium")
        self.base_device = base_device
        self.base_model = GPT2LMHeadModel.from_pretrained(gpt_model_path).to(base_device)
        self.base_tokenizer = GPT2TokenizerFast.from_pretrained(gpt_model_path)

        t5_model_path = get_model_path('t5-large')
        self.mask_device = mask_device
        self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_path).to(mask_device).half()
        self.mask_tokenizer = T5Tokenizer.from_pretrained(t5_model_path, model_max_length=512)
        self.mask_model.eval()

        print(self.base_model.device)
        print(self.mask_model.device)

        # hyper-para
        self.max_length = self.base_model.config.n_positions
        self.stride = 51
        self.mask_rate = 0.3

        self.span_length = 2
        self.perturb_pct = 0.3
        self.chunk_size = 100
        self.sample_num = sample_num
        self.threshold = threshold

        self.pattern = re.compile(r"<extra_id_\d+>")






        if cache_path and os.path.exists(cache_path):
            self.cache = load_from_pickle(cache_path)
        else:
            self.cache = {}

        self.cache_change = False
        self.cache_path = cache_path
        self.use_cache = use_cache

    @staticmethod
    def count_masks(texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    def __call__(self, text_list, disable_tqdm=True):
        ai_prob_list = []
        label_list = []

        text_list = tqdm(text_list, disable=disable_tqdm)

        for text in text_list:
            if self.use_cache and text in self.cache:
                ai_score = self.cache[text]
            else:
                ai_score = self.get_ai_score(text)
                self.cache[text] = ai_score
                self.cache_change = True


            if ai_score > self.threshold:
                cur_label = 1
            else:
                cur_label = 0

            ai_prob_list.append(ai_score)
            label_list.append(cur_label)

            text_list.set_description(f'Prob.: {np.mean(ai_prob_list):.4f}')

        return ai_prob_list, label_list


    def get_threshold(self):
        return self.threshold

    def save_cache(self):
        if self.cache_path and self.cache_change:
            save_to_pickle(self.cache, self.cache_path)

    def get_ai_score(self, text):
        # perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=0.3, mask_model=mask_model,
        #                                mask_tokenizer=mask_tokenizer)
        t1 = time.time()
        p_sampled_text = self.perturb_texts([text for _ in range(self.sample_num)])
        p_sequence_ll = self.get_lls(p_sampled_text)

        # p_sequence_ll = [_ll for _ll in p_sequence_ll if not (np.isnan(_ll) or np.isinf(_ll))]

        ll_std = np.std(p_sequence_ll)
        if ll_std == 0:
            ll_std = 1

        res_dict = {
        "original_ll": self.get_ll(text),
        "perturbed_ll": p_sequence_ll,
        "mean_perturbed_ll": np.mean(p_sequence_ll),
        "std_perturbed_ll": ll_std,
        }

        z_score = (res_dict['original_ll'] - res_dict['mean_perturbed_ll']) / res_dict['std_perturbed_ll']

        return z_score

    def perturb_texts(self, texts):
        outputs = []
        for i in range(0, len(texts), self.chunk_size):
            outputs.extend(self._perturb_texts(texts[i:i + self.chunk_size]))
        return outputs

    def _perturb_texts(self, texts):
        masked_texts = [self.tokenize_and_mask(x) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            # print(perturbed_texts)
            # print('-----------------------------')
            # print(masked_texts)
            # print('=============================')
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1

            if attempts > 300:
                break

        return perturbed_texts

    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.mask_device)
        outputs = self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=1.0, num_return_sequences=1,
                                      eos_token_id=stop_id)
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def tokenize_and_mask(self, text):
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        n_spans = self.perturb_pct * len(tokens) / (self.span_length + 2)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - self.span_length)
            end = start + self.span_length
            search_start = max(0, start - 1)
            search_end = min(len(tokens), end + 1)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def get_ll(self, text):
        if isinstance(self.base_model, str) and self.base_model.startswith("text-davinci"):
            kwargs = {"engine": self.base_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
            r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
            result = r['choices'][0]
            tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

            assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

            return np.mean(logprobs)
        else:
            with torch.no_grad():
                tokenized = self.base_tokenizer(text, return_tensors="pt").to(self.base_device)
                labels = tokenized.input_ids
                return - self.base_model(**tokenized, labels=labels).loss.item()

    def get_lls(self, texts):
        # if isinstance(base_model, str) and base_model.startswith("text-davinci"):
        #     pool = ThreadPool(30)
        #     return pool.map(self.get_ll, texts)
        # else:
        return [self.get_ll(text) for text in texts]


class OpenAIDetector(AIDetector):

    def __init__(self, cache_path=None):
        OPENAI_API_KEY = os.getenv(f"OPENAI_API_KEY")
        print(f'OPENAI KEY: {OPENAI_API_KEY[:10]}...')
        import openai
        openai.api_key = OPENAI_API_KEY


        self.official_classes = ['very unlikely', 'unlikely', 'unclear if it is', 'possibly', 'likely']
        self.binary_classes = {'very unlikely' : 0, 'unlikely': 0, 'unclear if it is': 1,
                               'possibly': 1, 'likely': 1}
        self.class_range = [0.1, 0.45, 0.9, 0.98]
        self.official_threshold = self.class_range[-2]

        print('Cache path', cache_path)

        if cache_path and os.path.exists(cache_path):
            print('Load cache from', cache_path)
            self.cache = load_from_pickle(cache_path)
        else:
            self.cache = {}

        self.cache_change = False
        self.cache_path = cache_path

    def get_threshold(self):
        return self.official_threshold

    def __call__(self, text_list, disable_tqdm=True):

        ai_prob_list = []
        label_list = []

        text_list = tqdm(text_list, disable=disable_tqdm)

        for text in text_list:
            prompt = text + "<|disc_score|>"
            # cache_updated = False
            if prompt in self.cache:
                top_logprobs = self.cache[prompt]
            else:
                while True:
                    try:
                        response = openai.Completion.create(engine="model-detect-v2",
                                                            prompt=prompt,
                                                            max_tokens=1,
                                                            temperature=1,
                                                            top_p=1,
                                                            n=1,
                                                            logprobs=5,
                                                            stop="\n",
                                                            stream=False)
                        top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
                        self.cache[prompt] = top_logprobs
                        self.cache_change = True
                        break
                    except openai.error.OpenAIError as e:
                        # If the API call fails, wait and retry after a delay
                        print("API error:", e)
                        print("Retrying in 5 seconds...")
                        time.sleep(5)

                # cache_updated = True

            if "\"" in top_logprobs:
                quote_logprob = np.exp(top_logprobs["\""])
            elif "!" in top_logprobs:
                quote_logprob = 1.0 - np.exp(top_logprobs["!"])
            else:
                print("No quote or exclamation mark found in top logprobs")
                quote_logprob = 0.5

            official_label = self.get_official_label(quote_logprob)
            binary_label = self.binary_classes[official_label]
            # print(official_label)

            ai_prob_list.append(quote_logprob)
            label_list.append(binary_label)
            text_list.set_description(f'Prob.: {np.mean(ai_prob_list):.4f}')


        return ai_prob_list, label_list


    def get_official_label(self, prob):

        class_index = next((i for i, x in enumerate(self.class_range) if x > prob), len(self.class_range))
        class_label = self.official_classes[class_index]

        return class_label

    def save_cache(self):
        if self.cache_path and self.cache_change:
            save_to_pickle(self.cache, self.cache_path)

# Credits for this code go to https://github.com/Haste171/gptzero
class GPTZeroDetector(AIDetector):

    def __init__(self, cache_path=None):
        self.base_url = 'https://api.gptzero.me/v2/predict'
        self.official_threshold = 0.65 # default 0.65, according to https://gptzero.me/faq

        if cache_path and os.path.exists(cache_path):
            self.cache = load_from_pickle(cache_path)
        else:
            self.cache = {}

        self.cache_change = False
        self.cache_path = cache_path

        GPTZERO_API_KEY = os.getenv(f"GPTZERO_API_KEY")
        print(f'OPENAI KEY: {GPTZERO_API_KEY[:10]}...')
        self.gptzero_api_key = GPTZERO_API_KEY


    def __call__(self, text_list, disable_tqdm=True):

        prob_list = []
        # responce_list = []
        label_list = []

        text_list = tqdm(text_list, disable=disable_tqdm)

        for text in text_list:
            if text in self.cache:
                prob_score = self.cache[text]
            else:
                self.cache_change = True
                url = f'{self.base_url}/text'
                headers = {
                    'accept': 'application/json',
                    'X-Api-Key': self.gptzero_api_key,
                    'Content-Type': 'application/json'
                }
                data = {
                    'document': text
                }
                while True:
                    response = requests.post(url, headers=headers, json=data).json()
                    if 'error' in response:
                        time.sleep(5)
                        print(response['error'])
                    else:
                        prob_score = response['documents'][0]['completely_generated_prob']
                        self.cache[text] = prob_score
                        break


            prob_list.append(prob_score)
            # responce_list.append(response)
            label_list.append(self.get_official_label(prob_score))
            text_list.set_description(f'Prob.: {np.mean(prob_list):.4f}')


        return prob_list, label_list

    def get_threshold(self):
        return self.official_threshold

    def get_official_label(self, prob):
        if prob < self.official_threshold:
            class_label = 0
        else:
            class_label = 1

        return class_label

    def save_cache(self):
        if self.cache_path and self.cache_change:
            print('Save cache to', self.cache_path)
            save_to_pickle(self.cache, self.cache_path)


class RankDetector(AIDetector):

    def __init__(self, base_device, log_rank=True):
        gpt_model_path = "gpt2-medium"
        self.base_device = base_device
        self.base_model = GPT2LMHeadModel.from_pretrained(gpt_model_path).to(base_device)
        self.base_tokenizer = GPT2TokenizerFast.from_pretrained(gpt_model_path)
        self.base_model.eval()

        self.log_rank = log_rank

    def get_rank_onetext(self, text):
        with torch.no_grad():
            tokenized = self.base_tokenizer(text, return_tensors="pt").to(self.base_device)
            logits = self.base_model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]

            # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True)
                       == labels.unsqueeze(-1)).nonzero()

            assert matches.shape[
                       1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float() + 1  # convert to 1-indexed rank
            if self.log_rank:
                ranks = torch.log(ranks)

            return - ranks.float().mean().item()

    def __call__(self, text_list, disable_tqdm=True):

        text_list = tqdm(text_list, disable=disable_tqdm)
        ai_score_list = []
        label_list = []

        for text in text_list:
            cur_score = self.get_rank_onetext(text)
            cur_label = 1 if cur_score > self.get_threshold() else 0

            ai_score_list.append(cur_score)
            label_list.append(cur_label)

        return ai_score_list, label_list

    def save_cache(self):
        return

    def get_threshold(self):
        return - 1.4


