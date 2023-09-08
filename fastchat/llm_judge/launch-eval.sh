MODEL=$1
ID=$2
CUDA_VISIBLE_DEVICES=7 python gen_model_answer.py --model-path $MODEL --model-id $ID
python gen_judgment.py --model-list $ID --mode pairwise-baseline --baseline-model vicuna-7b-v1.5
python gen_judgment.py --model-list $ID 
