import argparse

from sico.incontext_trainer import SICOTrainer
from my_utils.my_dataloader import load_init_data_list
import shared_dir


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
    parser.add_argument('--train-iter', type=int, default=8)
    parser.add_argument('--para-num', type=int, default=8)

    # parser.add_argument('--debug', action='store_ture')

    args = parser.parse_args()

    detector_name = args.detector
    dataset = args.dataset
    llm_name = args.llm
    task_type = args.task

    if task_type == 'essay':
        assert dataset == 'squad'
    elif task_type == 'qa':
        assert dataset == 'eli5'
    elif task_type == 'rev-gen':
        assert dataset == 'yelp'

    # hyper paras
    eval_size = args.eval_size
    max_train_iter = args.train_iter
    incontext_size = args.incontext_size
    para_num = args.para_num

    llm_gen_kwargs = {'temperature': 1}

    init_data_list = load_init_data_list(dataset_name=dataset, task_name=task_type, ic_size=incontext_size)# (x_ic, y_human, y_ai)
    tag = f'run_{dataset}_{task_type}_{llm_name}_{detector_name}_eval={eval_size}_ic={incontext_size}_iter={max_train_iter}'

    trainer = SICOTrainer(
        dataset=dataset,
        llm_name=llm_name,
        detector_name=detector_name,
        task_type=task_type,
        eval_size=eval_size,
        ic_num=incontext_size,
        max_train_iter=max_train_iter,
        gen_kwargs=llm_gen_kwargs,
        para_num=para_num,
        save_dir=shared_dir.train_result_dir,
        tag=tag,
        seed=5050,
        disable_wandb=False
    )

    trainer.train(init_data_list)