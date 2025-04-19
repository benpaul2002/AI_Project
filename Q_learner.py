import nltk
import textstat
import random
import numpy as np
import pickle

nltk.download('punkt_tab')

class Q_Learner():
    def __init__(self, models, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, load_q_table=False):
        self.models = models
        # With 3 complexity levels and 3 length levels
        complexity_levels = ["simple", "moderate", "complex"]
        length_levels = ["short", "medium", "long"]
        self.num_models = len(models)

        # Initialize Q-table
        if load_q_table:
            with open('q_table.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
        else:
            q_table = {}
            for complexity in complexity_levels:
                for length in length_levels:
                    q_table[(complexity, length)] = [0] * self.num_models
        self.q_table = q_table

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate

        self.stats = {"cheap_model_uses": 0, "expensive_model_uses": 0, "rewards": []}

    def choose_model(self, state):
        # Epsilon-greedy policy for exploration vs exploitation
        if random.random() < self.epsilon:
            # Exploration: randomly choose a model
            model_index = random.randint(0, len(self.models) - 1)
        else:
            # Exploitation: choose model with highest Q-value
            model_index = np.argmax(self.q_table[state])
        
        # Track usage statistics
        if model_index == 0:
            self.stats["cheap_model_uses"] += 1
        else:
            self.stats["expensive_model_uses"] += 1
            
        return model_index, self.models[model_index]
    
    def decay_epsilon(self, decay_rate=0.995):
        self.epsilon *= decay_rate
        self.epsilon = max(self.epsilon, 0.01)  # Minimum exploration rate
    
    def calculate_reward(self, model_index, is_correct):
        model_cost = self.models[model_index]['cost']
        # Base reward: 1 for correct, 0 for incorrect
        performance_score = 1 if is_correct else 0
        # Higher cost penalty for incorrect answers makes sense
        cost_factor = 0.05 if is_correct else 0.1
        reward = performance_score - (cost_factor * model_cost)
        self.stats["rewards"].append(reward)
        return reward
    
    def update_q_value(self, state, action, reward, next_state):
        # Standard Q-learning update formula
        current_q = self.q_table[state][action]
        best_next_q = max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q - current_q
        )
        
        self.q_table[state][action] = new_q

    def calculate_complexity(self, text):
        # Option 1: Readability metrics
        readability_score = textstat.flesch_kincaid_grade(text)
        
        # Option 2: Vocabulary diversity
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        lexical_diversity = unique_words / total_words
        
        # Option 3: Sentence complexity
        avg_sentence_length = sum(len(s.split()) for s in nltk.sent_tokenize(text)) / len(nltk.sent_tokenize(text))
        
        # Combined score (example)
        return readability_score + (lexical_diversity * 50) + (avg_sentence_length * 2)

    def get_state(self, text):
        # Length discretization
        num_words = len(text.split())
        if num_words < 50:
            length = "short"
        elif num_words < 200:
            length = "medium"
        else:
            length = "long"
        
        # Complexity discretization (example using readability)
        complexity_score = self.calculate_complexity(text)
        if complexity_score < 30:
            complexity = "simple"
        elif complexity_score < 70:
            complexity = "moderate" 
        else:
            complexity = "complex"
        
        return (complexity, length)  # State tuple
    
    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
