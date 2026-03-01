#!/bin/bash

SAVE_PATH="./experiment/Gemma_2b/"

python3 eval_gsm8k.py \
    --model $SAVE_PATH \
    --data_file ./data/test/GSM8K_test.jsonl \
    --batch_size 400

python3 eval_math.py \
    --model $SAVE_PATH \
    --data_file ./data/test/MATH_test.jsonl \
    --batch_size 400
