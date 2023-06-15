import numpy as np
import pathlib
from sklearn.metrics import roc_auc_score, roc_curve

from detectors import RoBERTaAIDetector, GPTZeroDetector, OpenAIDetector, DetectGPT, RankDetector
import shared_dir



def get_generated_text_saving_dir(dataset, attack_method):
    'outputs/test_results/[dataset]/[attack method]/generated_text.tsv'
    save_folder = pathlib.Path(shared_dir.test_results_dir, dataset, attack_method)
    save_folder.mkdir(parents=True, exist_ok=True)
    return save_folder.joinpath('generated_text.tsv')


def find_threshold(ai_prob_list, desired_rate=0.01):
    # Step 1: Sort the list of numbers in descending order
    sorted_numbers = sorted(ai_prob_list, reverse=True)

    # Step 2: Calculate the index that corresponds to the percentile
    index = len(sorted_numbers) * desired_rate

    # Step 3: Find the threshold
    if index.is_integer():
        if int(index) > 0:
            threshold = (sorted_numbers[int(index)] + sorted_numbers[int(index) - 1]) / 2
        else:
            threshold = sorted_numbers[int(index)]
    else:
        threshold = sorted_numbers[int(index)]

    return threshold


def get_ai_perc_threshold(ai_prob_list, threshold: float):
    assert len(ai_prob_list) > 0

    if not isinstance(ai_prob_list, np.ndarray):
        ai_prob_list = np.array(ai_prob_list)

    ai_num = np.sum(ai_prob_list > threshold)
    ai_perc = ai_num / len(ai_prob_list)

    # print(f'AI Percentage {ai_num} / {len(ai_prob_list)} = {ai_perc:.2%} with threshold {threshold:.5f}')
    return ai_perc

def get_score_saving_dir_arg(args):
    save_folder = pathlib.Path(shared_dir.test_results_dir, args.dataset, args.attacker)
    save_folder.mkdir(parents=True, exist_ok=True)
    save_dir = save_folder.joinpath(args.detector + '_score.tsv')

    return save_folder, save_dir

def get_score_saving_dir(dataset, gen_method, detector):
    save_folder = pathlib.Path(shared_dir.test_results_dir, dataset, gen_method)
    save_folder.mkdir(parents=True, exist_ok=True)
    save_dir = save_folder.joinpath(detector + '_score.tsv')

    return save_folder, save_dir


def get_ai_probs(text_list, detector_name, dataset, device, detector_cache=True):

    if detector_name in {'chatdetect', 'gpt2detect'}:
        ## Chat Detect

        detector = RoBERTaAIDetector(name=detector_name, device=device)
        # ai_prob_list, pred_list = detector.get_scores(text_list)
        # return ai_prob_list
    elif detector_name == 'gptzero':
        ## GPT-zero
        detector = GPTZeroDetector(
            cache_path=shared_dir.cache_folder_dir + f'detector/gptzero_{dataset}.pkl')
    elif detector_name == 'openai':
        detector = OpenAIDetector(
            cache_path=shared_dir.cache_folder_dir + f'detector/openai_{dataset}.pkl')
    elif detector_name == 'detectgpt':
        detector = DetectGPT(sample_num=100, base_device=device, mask_device=device,
                             cache_path=shared_dir.cache_folder_dir + f'detector/detectgpt_{dataset}.pkl',
                             use_cache=detector_cache)

    elif detector_name == 'logrank':
        detector = RankDetector(base_device=device, log_rank=True)
    else:
        raise Exception('Wrong detector', detector_name)


    ai_prob_list, pred_list = detector(text_list, disable_tqdm=False)
    detector.save_cache()

    _acc = (np.ones(len(text_list)) == np.array(pred_list)).sum() / len(pred_list)
    print(f'{_acc:.2%} of texts are detected as AI-gen with default threshold {detector.get_threshold():.2f}')

    return ai_prob_list


def get_auc(y_true, y_score):
    # y_true = np.ones(len(y_score))
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    return auc


