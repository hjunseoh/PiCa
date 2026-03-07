

export MODEL_PATH='google/gemma-2b'
export MASTER_ADDR="localhost"
export MASTER_PORT=$((RANDOM % 10000 + 10000))
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

SAVE_PATH="./experiment/Gemma_2b/"

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=1 --use_env train_math.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "./data/train/MetaMathQA-40K.json" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 999999999 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --adapter_name "pica" \
    --rank 256
