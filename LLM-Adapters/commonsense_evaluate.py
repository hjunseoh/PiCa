# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
import json
import os
import re
import sys
import argparse

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    raise ValueError("This script requires CUDA support for evaluation.")

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    share_gradio: bool = False,
):
    args = parse_args()

    def evaluate(
        instructions,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=32,
        **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        return outputs

    save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    if args.adapter in ["LoRA", "DoRA"] and args.lora_weights != "None":
        print("Merging LoRA/DoRA weights into the original weights")
        key_list = [(key, module) for key, module in model.named_modules()]
        for key, module in key_list:
            if hasattr(module, 'merge_weights'):
                print(f"Found mergable module: {key}")
                module.merge_weights = True
                module.train(mode=False)
    else:
        print("Evaluating with base model only, no adapter weights applied.")

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            print(data["instruction"])
            print(output)
            print('prediction:', predict)
            print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """

def load_data(args) -> list:
    file_path = f'/home/hjunseoh/DoRA/commonsense_reasoning/dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset) // batch_size if len(dataset) % batch_size == 0 else len(dataset) // batch_size + 1
    for i in range(num_batch):
        batch = dataset[i * batch_size: min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"], required=True)
    parser.add_argument('--model', choices=['google/gemma-2b', 'LLaMA-7B', "LLaMA-13B", 'LLaMA2-7B', 'LLaMA3-8B', 'LLaMA3.2-1B', 'LLaMA3.2-3B', 'Gemma-2B','Gemma-7B'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'DoRA', 'None', 'pica'], required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    return parser.parse_args()

def load_model(args) -> tuple:
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'Cannot find base model name by the value: {args.model}')
    
    lora_weights = args.lora_weights
    load_8bit = args.load_8bit

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0  

    print(f"Loading base model from: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_weights and lora_weights != "None":
        if args.adapter.lower() == 'pica':
            from pica import load_pica_model
            print(f"Loading PiCa adapter from: {lora_weights}")
            model = load_pica_model(model, lora_weights)
            model = model.merge_and_unload()
        else:
            print(f"Loading LoRA weights from: {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"": 0}
            )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model

def extract_answer(args, sentence: str) -> str:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

if __name__ == "__main__":
    main()
