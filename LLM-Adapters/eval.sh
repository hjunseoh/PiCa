#!/bin/bash

SAVE_PATH="./models/Gemma_2b_PiCa/"
DATASETS=("boolq" "piqa" "social_i_qa" "hellaswag" "winogrande" "ARC-Easy" "ARC-Challenge" "openbookqa")

for DATASET in "${DATASETS[@]}"; do
    echo "Evaluating dataset: ${DATASET}"
    python commonsense_evaluate.py \
      --model "Gemma-2B" \
      --adapter "pica" \
      --dataset "${DATASET}" \
      --base_model "google/gemma-2b" \
      --batch_size 4 \
      --lora_weights "${SAVE_PATH}" | tee -a "./evaluate/pica/gemma-2b/${DATASET}.txt"
    echo "Finished evaluating ${DATASET}"
    echo "--------------------------------"
done