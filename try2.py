from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
import re
import numpy as np
from tqdm import tqdm
import torch
from Q_learner import Q_Learner
from dqn import DQN_Learner

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
            'cost': 5  # Lower cost since it's a smaller model
        }
    
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
    model_obj = models[model_index]['model']
    tokenizer = models[model_index].get('tokenizer', None)
    if tokenizer:
        tokenizer = models[model_index]['tokenizer']
    
    # if models[model_index]['model_name'] == "wizardmath":
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model_obj.generate(
        inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
        attention_mask=inputs.attention_mask,
        # pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_q_learner(dataset, models):
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


def run_dqn(dataset, models):
    dqn_learner = DQN_Learner(
        models, 
        learning_rate=0.001,
        discount_factor=0.9,
        epsilon=0.1,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    dataset = gsm8k_dataset['train']
    for i in tqdm(range(5), desc="Training DQN learner"):
        current_problem = dataset[i]
        next_problem = dataset[i+1]
        
        # Get current state
        current_state = dqn_learner.get_state(current_problem["question"])
        
        # Choose model using the DQN policy
        model_index, model = dqn_learner.choose_model(current_state)
        
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
        reward = dqn_learner.calculate_reward(model_index, is_correct)
        
        # Get next state
        next_state = dqn_learner.get_state(next_problem["question"])
        
        # Train the DQN model
        dqn_learner.train(current_state, model_index, reward, next_state, done=False)
        
        # Decay epsilon after each problem
        dqn_learner.decay_epsilon()
    
    # Save the trained DQN model
    dqn_learner.save_model('dqn_model.pth')
    
    # Evaluation phase (optional)
    print("\nEvaluation Phase:")
    test_dataset = gsm8k_dataset['test']
    correct_predictions = 0
    total_predictions = min(10, len(test_dataset))  # Evaluate on first 10 test problems
    
    for i in range(total_predictions):
        problem = test_dataset[i]
        state = dqn_learner.get_state(problem["question"])
        
        # Use the trained policy with no exploration (epsilon=0)
        epsilon_backup = dqn_learner.epsilon
        dqn_learner.epsilon = 0
        model_index, _ = dqn_learner.choose_model(state)
        dqn_learner.epsilon = epsilon_backup
        
        model_output = process_problem(problem, model_index, models)
        predicted_answer = extract_answer(model_output)
        true_answer = extract_answer(problem["answer"])
        
        if predicted_answer == true_answer:
            correct_predictions += 1
            
        print(f"Test Problem {i}: {'Correct' if predicted_answer == true_answer else 'Incorrect'}")
        
    print(f"Accuracy: {correct_predictions/total_predictions:.2f}")
    print(f"Model usage statistics: Cheap model: {dqn_learner.stats['cheap_model_uses']}, Expensive model: {dqn_learner.stats['expensive_model_uses']}")

if __name__ == "__main__":
    gsm8k_dataset = {
        'train': get_dataset()[0],
        'test': get_dataset()[1]
    }

    models = [get_model('wizardmath')]

    run_dqn(gsm8k_dataset, models)
