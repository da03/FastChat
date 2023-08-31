export BATCH_SIZE=4
export ACC=8
export SAVE=output_vicuna_4096_b${BATCH_SIZE}_a${ACC}_split_4096_gpt_and_user
mkdir $SAVE
torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --data_path data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank_add_user_eos.json \
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
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --gpt_and_user True
CUDA_VISIBLE_DEVICES=0 python fastchat/train/train_mem.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf  \
    --data_path data_aug30/vicuna_format_dialogues-all_filtered_double_turns_chat_removeblank_add_user_eos.json \
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
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --gpt_and_user True
