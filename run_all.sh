export BATCH_SIZE=4
export ACC=8
export SAVE=output_vicuna_2048_b${BATCH_SIZE}_a${ACC}_nosplit
mkdir $SAVE
stdbuf -oL -eL torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --data_path data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank.json \
    --bf16 True \
    --output_dir $SAVE \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True > $SAVE/log.train.2048.nosplit 2>&1&

python -m fastchat.data.split_long_conversation --in data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat.json --out data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_split_2048.json --model-name meta-llama/Llama-2-7b-hf

CUDA_VISIBLE_DEVICES=0 python fastchat/train/train_mem.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --data_path data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat.json \
    --bf16 True \
    --output_dir output_vicuna \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True

python -m fastchat.data.split_long_conversation --in data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank.json --out data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank_split_2048.json --model-name meta-llama/Llama-2-7b-hf --max-length 2048

python remove_weird.py data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat.json data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank.json
python -m fastchat.data.split_long_conversation --in data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank.json --out ttt --model-name meta-llama/Llama-2-7b-hf --max-length 2048
python add_user_eos.py data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank.json data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank_add_user_eos.json
