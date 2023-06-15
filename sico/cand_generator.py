import os
import sys
# sys.path.append("/home/workspace/ChatGPT_detect/")
import nltk
import lemminflect

import numpy as np
from nltk.corpus import wordnet as wn
# from .parrot.parrot import Parrot
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy.stats import gmean

from my_utils.model_path import get_model_path
import re


wnl = WordNetLemmatizer()

def _split_last_punctuation(word):
    match = re.search(r'([\W_]+)$', word)
    if match:
        punctuation = match.group(1)
        clean_word = word[:-len(punctuation)]
        return clean_word, punctuation
    else:
        return word, ''


def clean_word_list_last_punct(word_list):
    return list(zip(*[_split_last_punctuation(w) for w in word_list]))


def align_format(orig_word, other_word_list):
    if orig_word.isupper():
        return [w.upper() for w in other_word_list]
    elif orig_word.islower():
        return [w.lower() for w in other_word_list]
    elif orig_word.istitle():
        return [w.title() for w in other_word_list]
    else:
        return other_word_list

    # if format_type == "upper":
    #     return word.upper()
    # elif format_type == "lower":
    #     return word.lower()
    # elif format_type == "title":
    #     return word.title()
    # else:
    #     return word


def _get_pos(sent, tagset='universal'):
    '''
    :param sent: list of word strings
    tagset: {'universal', 'default'}
    :return: list of pos tags.
    Universal (Coarse) Pos tags has  12 categories
        - NOUN (nouns)
        - VERB (verbs)
        - ADJ (adjectives)
        - ADV (adverbs)
        - PRON (pronouns)
        - DET (determiners and articles)
        - ADP (prepositions and postpositions)
        - NUM (numerals)
        - CONJ (conjunctions)
        - PRT (particles)
        - . (punctuation marks)
        - X (a catch-all for other categories such as abbreviations or foreign words)
    '''
    if tagset == 'default':
        word_n_pos_list = nltk.pos_tag(sent)
    elif tagset == 'universal':
        word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
    _, pos_list = zip(*word_n_pos_list)
    return pos_list


def pos_filter(ori_pos, new_pos_list):
    same = [True if ori_pos == new_pos or ({ori_pos, new_pos} <= {'noun', 'verb'})
            else False for new_pos in new_pos_list]
    return same


def valid_pos(orig_pos):
    return orig_pos in {'NOUN', 'VERB', 'ADJ', 'ADV'}


def _get_wordnet_pos(universal_pos):
    '''Wordnet POS tag'''
    # pos = spacy_token.tag_[0].lower()
    # if pos in ['r', 'n', 'v']:  # adv, noun, verb
    #     return pos
    # elif pos == 'j':
    #     return 'a'  # adj
    d = {
        'NOUN': 'n',
        'VERB': 'v',
        'ADJ': 'a',
        'ADV': 'v'
    }
    if universal_pos in d:
        return d[universal_pos]
    else:
        print(universal_pos, 'not valid')
        return None

def _generate_synonyms_wordnet(word, universal_pos, ptb_pos):
    synonyms = []
    candidates = []
    candidate_set = set()
    wordnet_pos = _get_wordnet_pos(universal_pos)  # 'r', 'a', 'n', 'v' or None

    lemmas = lemminflect.getAllLemmas(word)
    if lemmas and universal_pos in lemmas:
        final_lemmas = lemmas[universal_pos]
    else:
        final_lemmas = [wnl.lemmatize(word, wordnet_pos)]

    for orig_lemma in final_lemmas:
        # wordnet
        assert wordnet_pos is not None
        wordnet_synonym_lemmas = []

        synsets = wn.synsets(orig_lemma, pos=wordnet_pos)
        for synset in synsets:
            wordnet_synonym_lemmas.extend(synset.lemmas())


        for wordnet_synonym_lemma in wordnet_synonym_lemmas:
            synonym_lemma = wordnet_synonym_lemma.name().replace('_', ' ')

            if synonym_lemma.lower() == orig_lemma.lower() or len(synonym_lemma.split()) > 1:
                # the synonym produced is a phrase
                continue

            # delemma
            morph_synonyms = lemminflect.getInflection(synonym_lemma, tag=ptb_pos)

            synonyms.extend(morph_synonyms)

    # synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)


    for _, synonym in enumerate(synonyms):
        candidate_word = synonym
        if candidate_word in candidate_set:  # avoid repetition
            continue
        candidate_set.add(candidate_word)
        candidates.append(candidate_word)


    return candidates


def _wordnet_preprocess_onetext(word_list):
    idx_word_perturb_list = []
    sub_words_dict = {}

    orig_text = word_list.copy()
    len_text = len(word_list)
    orig_universal_pos_list = _get_pos(orig_text)
    orig_ptb_pos_list = _get_pos(orig_text, 'default')

    # word list to substring list
    cln_word_list, punct_list = clean_word_list_last_punct(word_list)

    # get synonyms for legal substrings
    synonyms_of_cln_word = {}
    for w_idx in range(len(cln_word_list)):
        cln_word = cln_word_list[w_idx]
        universal_pos = orig_universal_pos_list[w_idx]
        ptb_pos = orig_ptb_pos_list[w_idx]

        if not valid_pos(universal_pos):
            synonyms_of_cln_word[w_idx] = []
        else:
            synonyms_of_cln_word[w_idx] = _generate_synonyms_wordnet(cln_word, universal_pos, ptb_pos)





    # for w_idx, orig_word in enumerate(word_list):
    #
    #     cur_cln_synonyms = synonyms_of_cln_word[w_idx]
    #     if len(cur_cln_synonyms) == 0:
    #         continue
    #
    #     cur_cln_synonyms = align_format(orig_word, cur_cln_synonyms)
    #     idx_word_perturb_list.append((w_idx, orig_word))
    #     sub_words_dict[(w_idx, orig_word)] = cur_cln_synonyms
    #
    # # filter
    # filter.do_filtering(word_list, idx_word_perturb_list, sub_words_dict)
    #
    # # add punctuation
    # for w_idx, orig_word in enumerate(word_list):
    #     orig_punct = punct_list[w_idx]
    #     final_synonyms_of_word = [cln_syno + orig_punct for cln_syno in cur_cln_synonyms]



    for w_idx, orig_word in enumerate(word_list):

        cur_cln_synonyms = synonyms_of_cln_word[w_idx]
        if len(cur_cln_synonyms) == 0:
            continue

        cur_cln_synonyms = align_format(orig_word, cur_cln_synonyms)
        orig_punct = punct_list[w_idx]
        final_synonyms_of_word = [cln_syno + orig_punct for cln_syno in cur_cln_synonyms]

        if len(final_synonyms_of_word) > 0:
            idx_word_perturb_list.append((w_idx, orig_word))
            sub_words_dict[(w_idx, orig_word)] = final_synonyms_of_word

    return idx_word_perturb_list, sub_words_dict


class WordCandGenerator:

    def __init__(self):
        self.cache_ = {}


    def generate_cand_dict(self, input_str):
        word_list = input_str.split()
        key = input_str
        if key in self.cache_:
            # print('Use cache by key:', key)
            return self.cache_[key]
        else:

            ret = self._generate_cand_dict_wordlist(word_list)

            self.cache_[key] = ret
            return ret

    def _generate_cand_dict_wordlist(self, word_list):
        raise NotImplementedError


class ParaCandGenerator:

    def generate_para_dict(self):
        raise NotImplementedError


def strip_BPE_artifacts(token, model_type):
    """Strip characters such as "Ġ" that are left over from BPE tokenization.

    Args:
        token (str)
        model_type (str): type of model (options: "bert", "roberta", "xlnet")
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return token.replace("##", "")
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "longformer"]:
        return token.replace("Ġ", "")
    elif model_type == "xlnet":
        if len(token) > 1 and token[0] == "_":
            return token[1:]
        else:
            return token
    else:
        return token

def check_if_subword(token, model_type, starting=False):
    """Check if ``token`` is a subword token that is not a standalone word.

    Args:
        token (str): token to check.
        model_type (str): type of model (options: "bert", "roberta", "xlnet").
        starting (bool): Should be set ``True`` if this token is the starting token of the overall text.
            This matters because models like RoBERTa does not add "Ġ" to beginning token.
    Returns:
        (bool): ``True`` if ``token`` is a subword token.
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return True if "##" in token else False
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "longformer"]:
        if starting:
            return False
        else:
            return False if token[0] == "Ġ" else True
    elif model_type == "xlnet":
        return False if token[0] == "_" else True
    else:
        return False


class MLMFilter:

    def __init__(self, mlm_name='roberta-base', device='cuda:0', max_length=512):

        self.mask_lm = AutoModelForMaskedLM.from_pretrained(
                mlm_name
            )

        self.mlm_tokenizer = AutoTokenizer.from_pretrained(
                mlm_name, use_fast=True
            )

        self.mask_lm.to(device)
        self.mask_lm.eval()
        self.device = device
        self.model_type = self.mask_lm.config.model_type

        self.max_length = max_length
        self.batch_size = 16

    def _encode_text(self, text):
        """Encodes ``text`` using an ``AutoTokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self.mlm_tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoding.to(self.device)

    def flatten_replace(self, lst, index, replacement_list):
        ret_list = []
        for i in range(len(lst)):
            if i == index:
                ret_list.extend(replacement_list)
            else:
                ret_list.append(lst[i])

        return ret_list

    def do_filtering(self, word_list, idx_word_list, word_dict, mlm_conf_threshold=2e-4, max_mask_num=3):
        new_word_dict = {}
        new_idx_word_list = []

        # Prepare masked sentences
        masked_texts = []
        idx_maskn2maskidx = {} #(idx, mask_number) -> mask predictions index


        punct_str_dict = {}


        for word_idx, orig_word in idx_word_list: # for each position
            _, punct_str = _split_last_punctuation(orig_word)
            punct_str_dict[(word_idx, orig_word)] = punct_str
            for mask_n in range(1, max_mask_num + 1):
                idx_maskn2maskidx[(word_idx, mask_n)] = len(masked_texts)


                # build mask texts
                _masked_word_list = word_list.copy()
                replacement_list = [self.mlm_tokenizer.mask_token] * mask_n
                if punct_str:
                    replacement_list += [punct_str]
                _masked_word_list = self.flatten_replace(_masked_word_list, word_idx, replacement_list)

                masked_texts.append(' '.join(_masked_word_list))

        preds_list = []
        masked_index_list = []

        b_i = 0
        while b_i < len(masked_texts):
            b_inputs = self._encode_text(masked_texts[b_i: b_i + self.batch_size])
            with torch.no_grad():
                 b_preds = self.mask_lm(**b_inputs)[0] # (batch, token, vocab)
            preds_list.append(b_preds)

            b_ids = b_inputs["input_ids"].tolist()
            for i_inbatch in range(len(b_ids)): # for each text in batch
                try:
                    # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
                    masked_index = b_ids[i_inbatch].index(self.mlm_tokenizer.mask_token_id) # find mask token position
                    masked_index_list.append(masked_index)
                except ValueError:
                    masked_index_list.append(None)
                    continue
            b_i += self.batch_size

        all_preds = torch.cat(preds_list, dim=0)

        desired_probs = []
        for word_idx, orig_word in idx_word_list: # for each position
            for mask_n in range(1, max_mask_num + 1):
                mask_idx = idx_maskn2maskidx[(word_idx, mask_n)]
                preds = all_preds[mask_idx]
                masked_index = masked_index_list[mask_idx]
                mask_token_logits = preds[masked_index:masked_index + mask_n, :]
                mask_token_probs = torch.softmax(mask_token_logits, dim=1)
                desired_probs.append(mask_token_probs)




        for word_idx, orig_word in idx_word_list:
            candidate_words = word_dict[(word_idx, orig_word)]
            filtered_cand_words = []
            for cand in candidate_words:

                orig_cand = cand

                if punct_str_dict[(word_idx, orig_word)]:
                    cand = cand[:-1]

                if word_idx == 0 and self.model_type in ["gpt", "gpt2", "roberta", "bart",
                                                             "longformer"]:  # not first token
                    cand_token_ids = self.mlm_tokenizer.encode(cand, add_special_tokens=False)
                else:
                    cand_token_ids = self.mlm_tokenizer.encode('A ' + cand, add_special_tokens=False)[1:]

                cand_len = len(cand_token_ids)

                if cand_len > max_mask_num:
                    continue

                mask_idx = idx_maskn2maskidx[(word_idx, cand_len)]
                mask_token_probs = desired_probs[mask_idx]

                cand_probs = [mask_token_probs[t_i, t_id].cpu().item() for t_i, t_id in enumerate(cand_token_ids)]
                gmean_prob = gmean(cand_probs)
                if gmean_prob > mlm_conf_threshold:
                    filtered_cand_words.append(orig_cand)

            if len(filtered_cand_words) > 0:
                new_idx_word_list.append((word_idx, orig_word))
                new_word_dict[(word_idx, orig_word)] = filtered_cand_words


        return new_idx_word_list, new_word_dict





class WordNetCandGenerator(WordCandGenerator):

    def __init__(self, do_filter=True, mlm_conf_threshold=1e-4, device='cuda:0'):
        super().__init__()

        self.do_filter = do_filter
        if self.do_filter:
            self.filter = MLMFilter(mlm_name=get_model_path("distilroberta-base"), device=device)
            self.mlm_conf_threshold = mlm_conf_threshold




    def _generate_cand_dict_wordlist(self, word_list):
        idx_word_list, cand_dict = _wordnet_preprocess_onetext(word_list)
        if self.do_filter:
            idx_word_list, cand_dict = self.filter.do_filtering(word_list, idx_word_list, cand_dict, self.mlm_conf_threshold)

        ret = (word_list, idx_word_list, cand_dict)
        return ret


class ParaLLMGenerator(ParaCandGenerator):
    def __init__(self, llm_api, generation_kwargs, para_num=10, length_limit=False):
        self.llm_api = llm_api
        self.para_num = para_num
        self.generation_kwargs = generation_kwargs
        # self.max_length = 32
        # self.paraphraser = Parrot(device)

        # self.split_str = ', '

        self.length_limit = length_limit
        self.cache = {}

    def generate_para_dict(self, doc_str, prompt_str):

        sentence_list = sent_tokenize(doc_str)
        final_sentence_list = []
        sent_idx2cand_sent = {}
        sent_idx_list = []

        def _is_short(sent_):
            return len(sent_.split()) < 5

        for s_idx in range(len(sentence_list)):
            # cur_max_length = sentence_list[s_idx]
            cur_sent = sentence_list[s_idx]

            if _is_short(cur_sent):
                if len(final_sentence_list) > 0:  # have previous, concatenate to previous
                    final_sentence_list[-1] = final_sentence_list[-1] + ' ' + cur_sent
                elif s_idx + 1 < len(sentence_list):  # concatenate to next string
                    sentence_list[s_idx + 1] = cur_sent + ' ' + sentence_list[s_idx + 1]
                else:
                    final_sentence_list.append(cur_sent)
            else:
                final_sentence_list.append(cur_sent)


        for s_idx in range(len(final_sentence_list)):
            # cur_max_length = sentence_list[s_idx]
            cur_sent = final_sentence_list[s_idx]
            cur_max_length = self._get_length(cur_sent)

            # paraphrased_cand_list = []
            # for _ in range(self.para_num):
            #     paraphrased_cand = self.llm_api(prompt_str.format(cur_sent),
            #                                          max_new_tokens=cur_max_length,
            #                                          return_num=1,
            #                                          generation_kwargs=self.generation_kwargs)
            #     paraphrased_cand_list.append(self.para_num)
            # paraphrased_cand_list = ['Test 1111'] * self.para_num

            paraphrased_cand_list = self.llm_api(prompt_str.format(cur_sent),
                                                     max_new_tokens=cur_max_length,
                                                     return_num=self.para_num,
                                                     generation_kwargs=self.generation_kwargs)

            if self.length_limit:
                orig_char_length = len(cur_sent)
                # filter by length, skip too short or too long
                paraphrased_cand_list = [para for para in paraphrased_cand_list if orig_char_length // 2 < len(para) < orig_char_length * 1.8]

                if len(paraphrased_cand_list) == 0:
                    print('=========================== Bad Prompt of Paraphrasing ============================\n',
                          prompt_str.format(cur_sent))

            # post process
            if self.llm_api.need_post_process:
                raise NotImplementedError

            # paraphrased_cand_list = ['this is a test'] * self.para_num

            if len(paraphrased_cand_list) > 0:
                k_ = (s_idx, cur_sent)
                sent_idx2cand_sent[k_] = paraphrased_cand_list
                sent_idx_list.append(k_)

        return final_sentence_list, sent_idx_list, sent_idx2cand_sent

    def _get_length(self, text):
        return int(len(text.split()) * 1.5)



if __name__ == '__main__':
    # c = CFCandGenerator(top_k=10, min_cossim=0.8)
    c = WordNetCandGenerator(mlm_conf_threshold=2e-5)

    t = '''Yo, there's a buncha types of trees n' stuff. Bushes: short, lots of stems. Shrubs: taller, defined. So, trees have only one main stem/trunk and can grow much larger than bushes and shrubs.'''


    part_list, idx_part_list, cand_dict = c.generate_cand_dict(t)

    for k in cand_dict:
        print(k)
        print(cand_dict[k])

