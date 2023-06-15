# =================== SICO-Para ==============================

python SICO_train.py --dataset squad --llm chatgpt --detector chatdetect --task paraphrase --incontext-size 8 --eval-size 32 --train-iter 8

python SICO_train.py --dataset eli5 --llm chatgpt --detector chatdetect --task paraphrase --incontext-size 8 --eval-size 32 --train-iter 8

python SICO_train.py --dataset yelp --llm chatgpt --detector chatdetect --task paraphrase --incontext-size 8 --eval-size 32 --train-iter 8

# =================== SICO-Gen ==============================

python SICO_train.py --dataset yelp --llm chatgpt --detector chatdetect --task rev-gen --incontext-size 8 --eval-size 32 --train-iter 8

python SICO_train.py --dataset squad --llm chatgpt --detector chatdetect --task essay --incontext-size 8 --eval-size 32 --train-iter 8

python SICO_train.py --dataset eli5 --llm chatgpt --detector chatdetect --task qa --incontext-size 8 --eval-size 32 --train-iter 8







