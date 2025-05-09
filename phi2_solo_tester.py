from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
import re
import numpy as np
from tqdm import tqdm
import torch
import random
import pickle
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import time

def get_dataset():
    train_dataset = load_dataset("openai/gsm8k", "main", split='train')
    test_dataset = load_dataset("openai/gsm8k", "main", split='test')
    return train_dataset, test_dataset

def get_model(model_name):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    if model_name == "wizardmath":
        wizardmath_tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardMath-7B-V1.1")
        wizardmath_model = AutoModelForCausalLM.from_pretrained(
            "WizardLM/WizardMath-7B-V1.1",
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.float16
        )
        return {
            'model': wizardmath_model,
            'model_name': "wizardmath",
            'tokenizer': wizardmath_tokenizer,
            'cost_per_token': 0.7
        }
    elif model_name == "phi2":
        phi2_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        phi2_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        return {
            'model': phi2_model,
            'model_name': "phi2",
            'tokenizer': phi2_tokenizer,
            'cost_per_token': 0.13
        }

def extract_answer(answer_text):
    # The final answer in GSM8K follows the '####' pattern
    match = re.search(r'####\s*(-?\d+)', answer_text)
    if match:
        return match.group(1).strip()
    return None

def process_problem(problem, model_index, models):
    prompt = f"""

Follow these instructions:
1. Work through the problem step by step
2. Calculate the numerical answer
3. On the last line, write ONLY: #### <numerical answer>. Do not add any units like "kg" or "m", or any currency symbols like "$".
4. Do not write anything after the final answer

-------------------
EXAMPLE FORMAT:
Step 1: [explanation]
Step 2: [explanation]
Final calculation: [calculation]
#### [numerical answer]
-------------------

NOW SOLVE THE PROBLEM CORRECTLY: {problem['question']}
"""
    # print("Entered global process problem")
    model_obj = models[model_index]['model']
    tokenizer = models[model_index].get('tokenizer', None)
    if tokenizer:
        tokenizer = models[model_index]['tokenizer']

    # if models[model_index]['model_name'] == "wizardmath":
    inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
    outputs = model_obj.generate(
        inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        attention_mask=inputs.attention_mask,
        # pad_token_id=tokenizer.eos_token_id,
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    prompt_end = full_output.find(f"NOW SOLVE THE PROBLEM CORRECTLY: {problem['question']}")
    if prompt_end != -1:
        prompt_end = prompt_end + len(f"NOW SOLVE THE PROBLEM CORRECTLY: {problem['question']}")
        model_response = full_output[prompt_end:].strip()
    else:
        model_response = full_output

    hash_match = re.search(r'####\s*([\$]?\s*\d+(?:\.\d+)?)', model_response)
    if hash_match:
        answer_text = hash_match.group(1)
        numeric_match = re.search(r'(\d+(?:\.\d+)?)', answer_text)
        if numeric_match:
            numeric_answer = numeric_match.group(1)
            # return f"{prompt}\n\n{model_response.split('####')[0].strip()}\n#### {numeric_answer}"
            return {
                'prompt': prompt,
                'response': model_response,
                'answer': numeric_answer
            }

    answer_match = re.search(r'(?:final answer|the answer is)[^0-9]*?([\$]?\s*\d+(?:\.\d+)?)',
                            model_response.lower())
    if answer_match:
        answer_text = answer_match.group(1)
        numeric_match = re.search(r'(\d+(?:\.\d+)?)', answer_text)
        if numeric_match:
            numeric_answer = numeric_match.group(1)
            answer_position = model_response.lower().find(answer_match.group(0))
            if answer_position != -1:
                # return f"{prompt}\n\n{model_response[:answer_position].strip()}\n#### {numeric_answer}"
                return {
                    'prompt': prompt,
                    'response': model_response,
                    'answer': numeric_answer
                }

    lines = model_response.split('\n')
    for i in range(len(lines)-1, max(0, len(lines)-5), -1):
        line = lines[i]
        if len(line.strip()) < 1 or any(word in line.lower() for word in ["step", "explanation"]):
            continue

        numeric_match = re.search(r'(\d+(?:\.\d+)?)', line)
        if numeric_match:
            numeric_answer = numeric_match.group(1)
            # return f"{prompt}\n\n{model_response.split(line)[0].strip()}\n#### {numeric_answer}"
            return {
                'prompt': prompt,
                'response': model_response,
                'answer': numeric_answer
            }

    # return full_output
    return {
        'prompt': prompt,
        'response': full_output,
        'answer': None
    }

temp_set = get_dataset()
gsm8k_dataset = {
    'train': temp_set[0],
    'test': temp_set[1]
}
# models = [get_model('phi2'), get_model('wizardmath')]
model = get_model('phi2')

num_problems = 500
start_idx = 2001
subset = gsm8k_dataset['train'].select(range(start_idx, start_idx+num_problems))
total_correct = 0
cur_problem_idx = 0
phi2_preds = []

start_time = time.time()
for problem in tqdm(subset):
    cur_problem_idx += 1
    # print(problem)
    correct_answer = extract_answer(problem['answer'])
    prediction = process_problem(problem, 0, [model])
    predicted_answer = prediction['answer']
    # print(f"Correct Answer: {correct_answer}")
    # print(f"Predicted Answer: {predicted_answer}")
    if predicted_answer is not None and float(predicted_answer) == float(correct_answer):
        total_correct += 1
        phi2_preds.append({'question': problem['question'], 'answer': problem['answer'], 'is_correct': True})
    else:
        phi2_preds.append({'question': problem['question'], 'answer': problem['answer'], 'is_correct': False})
    
end_time = time.time()
#     questions = [entry['question'] for entry in phi2_preds]
#     answers = [entry['answer'] for entry in phi2_preds]
#     correctness = [entry['is_correct'] for entry in phi2_preds]

# # Create a DataFrame
# problem_data = pd.DataFrame({
#     'question': questions,
#     'answer': answers,
#     'is_correct': correctness
# })

# Save to CSV
problem_data = pd.DataFrame(phi2_preds)
problem_data.to_csv('phi2_preds_solo.csv', index=False)

accuracy = total_correct / num_problems
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Time taken: {end_time - start_time:.2f} seconds")
