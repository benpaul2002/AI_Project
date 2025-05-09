import nltk
import textstat
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

nltk.download('punkt')

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN_Learner():
    def __init__(self, models, learning_rate=0.001, discount_factor=0.9, epsilon=0.1, 
                 epsilon_decay=0.995, epsilon_min=0.01, load_model=False):
        self.models = models
        self.num_models = len(models)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # State space: complexity (3 levels) and length (3 levels) one-hot encoded
        self.state_size = 6  # 3 complexity + 3 length
        self.action_size = self.num_models

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        if load_model:
            self.model.load_state_dict(torch.load('dqn_model.pth'))

        self.stats = {"cheap_model_uses": 0, "expensive_model_uses": 0, "rewards": []}

    def one_hot_state(self, state):
        complexity_levels = ["simple", "moderate", "complex"]
        length_levels = ["short", "medium", "long"]
        state_vec = np.zeros(self.state_size)
        state_vec[complexity_levels.index(state[0])] = 1
        state_vec[3 + length_levels.index(state[1])] = 1
        return torch.tensor(state_vec, dtype=torch.float32).to(self.device)

    def choose_model(self, state):
        if random.random() < self.epsilon:
            model_index = random.randint(0, self.num_models - 1)
        else:
            state_tensor = self.one_hot_state(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            model_index = torch.argmax(q_values).item()

        if model_index == 0:
            self.stats["cheap_model_uses"] += 1
        else:
            self.stats["expensive_model_uses"] += 1

        return model_index, self.models[model_index]

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def calculate_reward(self, model_index, is_correct):
        model_cost = self.models[model_index]['cost']
        performance_score = 1 if is_correct else 0
        cost_factor = 0.05 if is_correct else 0.1
        reward = performance_score - (cost_factor * model_cost)
        self.stats["rewards"].append(reward)
        return reward

    def train(self, state, action, reward, next_state, done=False):
        state_tensor = self.one_hot_state(state).unsqueeze(0)
        next_state_tensor = self.one_hot_state(next_state).unsqueeze(0)

        self.model.train()
        q_values = self.model(state_tensor)
        q_value = q_values[0, action]

        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            max_next_q_value = torch.max(next_q_values)
            target = reward + (self.discount_factor * max_next_q_value * (1 - int(done)))

        loss = self.criterion(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_complexity(self, text):
        readability_score = textstat.flesch_kincaid_grade(text)
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        lexical_diversity = unique_words / total_words
        avg_sentence_length = sum(len(s.split()) for s in nltk.sent_tokenize(text)) / len(nltk.sent_tokenize(text))
        return readability_score + (lexical_diversity * 50) + (avg_sentence_length * 2)

    def get_state(self, text):
        num_words = len(text.split())
        if num_words < 50:
            length = "short"
        elif num_words < 200:
            length = "medium"
        else:
            length = "long"

        complexity_score = self.calculate_complexity(text)
        if complexity_score < 30:
            complexity = "simple"
        elif complexity_score < 70:
            complexity = "moderate"
        else:
            complexity = "complex"

        return (complexity, length)

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
