from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import numpy as np
from tqdm import tqdm
from Q_learner import Q_Learner

def get_model(model_name):
    if model_name == "tinyllama":
        tinyllama_path = hf_hub_download(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        )
        tinyllama = Llama(
            model_path=tinyllama_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35  # Adjust based on your GPU capacity
        )
        return {
            'model': tinyllama,
            'cost': 1
        }
    elif model_name == "wizardmath":
        wizardmath_tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardMath-7B-V1.1")
        wizardmath_model = AutoModelForCausalLM.from_pretrained(
            "WizardLM/WizardMath-7B-V1.1",
            device_map="auto",
            load_in_4bit=True
        )
        return {
            'model': wizardmath_model,
            'tokenizer': wizardmath_tokenizer,
            'cost': 10
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
    prompt = f"Solve this math problem step by step: {problem['question']}"
    
    if model_index == 0:  # tinyllama
        model_obj = models[0]['model']
        result = model_obj.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.1,
            stop=["Q:", "\n\n"]
        )
        return result['choices'][0]['text']
    else:  # wizardmath
        model_obj = models[1]['model']
        tokenizer = models[1]['tokenizer']
        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        outputs = model_obj.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    gsm8k_dataset = {
        'train': get_dataset()[0],
        'test': get_dataset()[1]
    }

    # models = {
    #     'tinyllama': get_model('tinyllama'),
    #     'wizardmath': get_model('wizardmath')
    # }

    models = [get_model('tinyllama'), get_model('wizardmath')]

    q_learner = Q_Learner(models)

    dataset = gsm8k_dataset['train']
    for i in tqdm(range(len(dataset)-1), desc="Training Q-learner"):
        current_problem = dataset[i]
        next_problem = dataset[i+1]
        
        current_state = q_learner.get_state(current_problem["question"])
        next_state = q_learner.get_state(next_problem["question"])
        
        model_index, model = q_learner.choose_model(current_state)
        
        # Process the current problem
        model_output = process_problem(current_problem, model_index, models)
        
        # Extract answers and check correctness
        predicted_answer = extract_answer(model_output)
        true_answer = extract_answer(current_problem["answer"])
        is_correct = (predicted_answer == true_answer) if predicted_answer and true_answer else False
        
        # Calculate reward
        reward = q_learner.calculate_reward(model_index, is_correct)
        
        # Update Q-values
        q_learner.update_q_value(current_state, model_index, reward, next_state)
        
        # Decay epsilon after each problem
        q_learner.decay_epsilon()

    # Handle the last problem separately (terminal state)
    last_problem = dataset[-1]
    last_state = q_learner.get_state(last_problem["question"])
    model_index, model = q_learner.choose_model(last_state)

    # Process the last problem
    model_output = process_problem(last_problem, model_index, models)

    predicted_answer = extract_answer(model_output)
    true_answer = extract_answer(last_problem["answer"])
    is_correct = (predicted_answer == true_answer) if predicted_answer and true_answer else False

    # For terminal state, just update with immediate reward
    terminal_reward = q_learner.calculate_reward(model_index, is_correct)
    current_q = q_learner.q_table[last_state][model_index]
    new_q = current_q + q_learner.learning_rate * (terminal_reward - current_q)
    q_learner.q_table[last_state][model_index] = new_q

    # Print training statistics
    print(f"Training complete!")
    print(f"Final epsilon: {q_learner.epsilon:.4f}")
    print(f"Cheap model uses: {q_learner.stats['cheap_model_uses']}")
    print(f"Expensive model uses: {q_learner.stats['expensive_model_uses']}")
    print(f"Average reward: {np.mean(q_learner.stats['rewards']):.4f}")

    # Test model
    test_dataset = gsm8k_dataset['test']
    correct_predictions = 0
    total_predictions = len(test_dataset)
    for i in tqdm(range(total_predictions), desc="Testing Q-learner"):
        test_problem = test_dataset[i]
        test_state = q_learner.get_state(test_problem["question"])
        model_index, model = q_learner.choose_model(test_state)

        # Process the test problem
        model_output = process_problem(test_problem, model_index, models)

        predicted_answer = extract_answer(model_output)
        true_answer = extract_answer(test_problem["answer"])
        is_correct = (predicted_answer == true_answer) if predicted_answer and true_answer else False

        if is_correct:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy:.4f}")
