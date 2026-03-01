#!/bin/bash

DATA_PATH='./ft-training_set/commonsense_15k.json'
SAVE_PATH="./models/Gemma_2b/"

DATASETS=("boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Easy" "ARC-Challenge" "openbookqa")

for DATASET in "${DATASETS[@]}"; do
    echo "Evaluating dataset: ${DATASET}"
    python commonsense_evaluate.py \
      --model "Gemma-2B" \
      --adapter "None" \
      --dataset "${DATASET}" \
      --base_model "${SAVE_PATH}" \
      --batch_size 4 \
      --lora_weights "None" | tee -a "./evaluate/pica/gemma-2b/${DATASET}.txt"
    echo "Finished evaluating ${DATASET}"
    echo "--------------------------------"
done