#!/bin/bash

MODEL_PATH='google/gemma-2b'
DATA_PATH='./ft-training_set/commonsense_15k.json'
SAVE_PATH="./models/Gemma_2b/"

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3198 finetune.py \
  --base_model $MODEL_PATH \
  --data_path $DATA_PATH \
  --output_dir $SAVE_PATH \
  --batch_size 64 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-3 \
  --cutoff_len 512\
  --val_set_size 120 \
  --adapter_name pica \
  --lora_r 256 \
  --eval_step 200  --save_step 200 \
  --lora_target_modules "q_proj","v_proj","k_proj","up_proj","down_proj" \
  

