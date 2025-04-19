from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
import re
import numpy as np
from tqdm import tqdm
from Q_learner import Q_Learner
import torch

def get_model(model_name):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Match your input dtype
        bnb_4bit_quant_type="nf4",  # Add quantization type
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
            'cost': 10
        }
    # if model_name == "smolLM2":
    #     tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    #     model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    #     return {
    #         'model': model,
    #         'model_name': "smolLM2",
    #         'tokenizer': tokenizer,
    #         'cost': 4
    #     }
    
def get_dataset():
    train_dataset = load_dataset("openai/gsm8k", "main", split='train')
    test_dataset = load_dataset("openai/gsm8k", "main", split='test')
    return train_dataset, test_dataset

# Function to extract final answer from GSM8K format
def extract_answer(answer_text):
    # The final answer in GSM8K follows the '####' pattern
    match = re.search(r'####\s*(-?\d+)', answer_text)
    if match:
        return match.group(1).strip()
    return None

def process_problem(problem, model_index, models):
    prompt = f"""Solve ONLY this math problem step by step: 
{problem['question']}

Follow these instructions:
1. Work through the problem step by step
2. Calculate the final answer
3. On the last line, write ONLY: #### <your numerical answer>

-------------------
EXAMPLE FORMAT:
Step 1: [explanation]
Step 2: [explanation]
Final calculation: [calculation]
#### <answer>
-------------------

NOW SOLVE THE PROBLEM CORRECTLY:
"""
    model_obj = models[model_index]['model']
    tokenizer = models[model_index].get('tokenizer', None)
    if tokenizer:
        tokenizer = models[model_index]['tokenizer']
    
    if models[model_index]['model_name'] == "wizardmath":
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model_obj.generate(
            inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    # elif models[model_index]['model_name'] == "smolLM2":
    #     inputs = tokenizer(prompt, return_tensors="pt")
    
    #     # Special generation parameters for SmolLM2
    #     outputs = model_obj.generate(
    #         inputs.input_ids,
    #         max_new_tokens=1024,
    #         temperature=0.3,  # Better for this architecture
    #         top_p=0.9,        # Recommended for SmolLM
    #         do_sample=True,
    #         pad_token_id=tokenizer.eos_token_id,
    #         repetition_penalty=1.1  # Reduces answer duplication
    #     )
    #     # Skip prompt in response and clean output
    #     full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return full_response.replace(prompt, "").strip()

if __name__ == "__main__":
    gsm8k_dataset = {
        'train': get_dataset()[0],
        'test': get_dataset()[1]
    }

    models = [get_model('wizardmath')]

    q_learner = Q_Learner(models)

    dataset = gsm8k_dataset['train']
    for i in tqdm(range(5), desc="Training Q-learner"):
        current_problem = dataset[i]
        next_problem = dataset[i+1]
        
        current_state = q_learner.get_state(current_problem["question"])
        next_state = q_learner.get_state(next_problem["question"])
        
        model_index, model = q_learner.choose_model(current_state)
        
        # Process the current problem
        model_output = process_problem(current_problem, model_index, models)

        if model_index == 0:
            print(f"\nProblem {i}: {current_problem['question']}")
            print(f"\nChosen model: {model_index} ({'cheap' if model_index == 0 else 'expensive'})")
            print(f"\nModel output: {model_output}")
        
        # Extract answers and check correctness
        predicted_answer = extract_answer(model_output)
        print(f"Predicted answer: {predicted_answer}")
        true_answer = extract_answer(current_problem["answer"])
        print(f"True answer: {true_answer}")
        is_correct = (predicted_answer == true_answer) if predicted_answer and true_answer else False
        
        # Calculate reward
        reward = q_learner.calculate_reward(model_index, is_correct)
        
        # Update Q-values
        q_learner.update_q_value(current_state, model_index, reward, next_state)
        
        # Decay epsilon after each problem
        q_learner.decay_epsilon()

    # # Handle the last problem separately (terminal state)
    # last_problem = dataset[-1]
    # last_state = q_learner.get_state(last_problem["question"])
    # model_index, model = q_learner.choose_model(last_state)

    # # Process the last problem
    # model_output = process_problem(last_problem, model_index, models)

    # predicted_answer = extract_answer(model_output)
    # true_answer = extract_answer(last_problem["answer"])
    # is_correct = (predicted_answer == true_answer) if predicted_answer and true_answer else False

    # # For terminal state, just update with immediate reward
    # terminal_reward = q_learner.calculate_reward(model_index, is_correct)
    # current_q = q_learner.q_table[last_state][model_index]
    # new_q = current_q + q_learner.learning_rate * (terminal_reward - current_q)
    # q_learner.q_table[last_state][model_index] = new_q

    # # Print training statistics
    # print(f"Training complete!")
    # print(f"Final epsilon: {q_learner.epsilon:.4f}")
    # print(f"Cheap model uses: {q_learner.stats['cheap_model_uses']}")
    # print(f"Expensive model uses: {q_learner.stats['expensive_model_uses']}")
    # print(f"Average reward: {np.mean(q_learner.stats['rewards']):.4f}")

    # # Test model
    # test_dataset = gsm8k_dataset['test']
    # correct_predictions = 0
    # total_predictions = len(test_dataset)
    # for i in tqdm(range(1), desc="Testing Q-learner"):
    #     test_problem = test_dataset[i]
    #     test_state = q_learner.get_state(test_problem["question"])
    #     model_index, model = q_learner.choose_model(test_state)

    #     # Process the test problem
    #     model_output = process_problem(test_problem, model_index, models)

    #     predicted_answer = extract_answer(model_output)
    #     true_answer = extract_answer(test_problem["answer"])
    #     print(f"Predicted: {predicted_answer}, True: {true_answer}")
    #     is_correct = (predicted_answer == true_answer) if predicted_answer and true_answer else False

    #     if is_correct:
    #         correct_predictions += 1
    # accuracy = correct_predictions / total_predictions
    # print(f"Test Accuracy: {accuracy:.4f}")
