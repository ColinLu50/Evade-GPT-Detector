# Large Language Models can be Guided to Evade AI-Generated Text Detection

Source code for paper [*Large Language Models can be Guided to Evade AI-Generated Text Detection*](https://arxiv.org/abs/2305.10847).

We introduce **SICO**, a **S**ubstitution-based **I**n-**C**ontext example **O**ptimization method, 
which automatically build prompt that guide Large Language Models (LLMs), such as ChatGPT, to generate human-like texts. 

SICO successfully evades all exisiting AI-generation text detectors, including GPTzero and OpenAI official detector.


## Run SICO


### Requirements
Create environment by conda:

``
conda env create -f environment.yml
``

### LLM setup

- For `ChatGPT`, you need to set OpenAI API keys in the environment `export OPENAI_API_KEY={your OpenAI API key}`.
- For `Vicuna`, please follow the instruction [here](https://github.com/lm-sys/FastChat). And set the local url in `sico/LLM_api.py`.

### Detectors setup

- `GPTzero`: Set the GPTzero API key  in environment `export GPTZERO_API_KEY={your GPTzero API key}`. Key can be obtained from [GPTzero website](https://gptzero.me/).
- `OpenAI detector`: Set the OpenAI API in environment `export OPENAI_API_KEY={your OpenAI API key}`
- `ChatGPT detector, GPT2 detector, DetectGPT, Log-Rank`: Required models will be automatically download model from HuggingFace.

### Datasets
All datasets we used are listed in `datasets` folder, containing `squad,eli5,yelp`.
Each subfolder has three tsv files: `eval.tsv` for evaluation during training, `test.tsv` for final test, and `incontext.tsv` for initialization and building in-context examples.

### Run training

Run `SICO_train.py` to start training procedure.

Here we explain each required argument in details:

- `--llm`: Base LLM used for training, including `[chatgpt, vicuna]`
- `--dataset`: Dataset we use, including `[squad, eli5, yelp]`
- `--detector`: Proxy detector we use for training, including `['chatdetect', 'gpt2detect', 'gptzero', 'openai', 'detectgpt', 'logrank']`
- `--task`: The task type of training, including `['essay', 'qa', 'rev-gen', 'paraphrase']`. Notice that `paraphrase` task matches all dataset, but `essay, qa, rev-gen` tasks only match `squad, eli5, yelp`, respectively.
- `--incontext-size`: Size of in-context examples.
- `--eval-size`: Size of evaluation data during training.
- `--train-iter`:  Maximum training iteration.

#### Examples:

Reimplement of SICO-gen on essay writing task of SQuAD dataset:

```
python SICO_train.py 
    --dataset squad 
    --llm chatgpt 
    --detector chatdetect 
    --task essay
    --incontext-size 8
    --eval-size 32
    --train-iter 6
```

Reimplement of SICO-para for open-ended question answering task:
```
python SICO_train.py 
    --dataset yelp 
    --llm chatgpt 
    --detector chatdetect 
    --task paraphrase
    --incontext-size 8
    --eval-size 32
    --train-iter 6
```

After training, the optimized prompt is stored in `./outputs/results/` and training log is stored in `./outputs/logs/`.

### Run testing

Run `SICO_test_gen.py` to use trained-prompt to generate texts, the arguments are the same as `SICO_train.py`.
Extra parameter `--test-size` show the number of cases you want to test.

After running `SICO_test_gen.py`, the generated texts are stored in `test_results/{dataset}/SICO-{dataset}-{task}-{llm}-{proxy_detector}` folder. 
Then run `run_test_detection.py` to get the AI-generated probability from different detectors, which is stored in same folder, named `{test_detector}_score.tsv`.

#### Examples:

Test SICO-gen on essay writing task of SQuAD dataset, where the prompt is trained using ChatGPT and ChatGPT Detector:
```
python SICO_test_gen.py 
    --dataset squad 
    --llm chatgpt 
    --detector chatdetect 
    --task essay
    --incontext-size 8
    --eval-size 32
    --train-iter 6
    --test-size 100
```

Test the performance of optimized prompt against `DetectGPT detector`:

```
python run_test_detection.py 
    --dataset squad
    --method SICO-squad-essay-chatgpt-chatdetect 
    --detector detectgpt
```

## To-Do List
Here is our planned roadmap for future developments:
- [x] **Open Source Code**
- [ ] **Share Effective Prompts**
- [ ] **Share Benchmark**

Stay tuned for further updates and developments on these tasks. We encourage community engagement and welcome any form of feedback or contributions to our project.