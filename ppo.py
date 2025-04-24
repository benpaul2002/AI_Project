import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import nltk
import textstat

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPO_Agent():
    def __init__(self, models, state_dim=6, learning_rate=0.001, gamma=0.99, 
                 clip_epsilon=0.2, ppo_epochs=4, batch_size=32):
        self.models = models
        self.num_models = len(models)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_network = PolicyNetwork(state_dim, self.num_models).to(self.device)
        self.value_network = ValueNetwork(state_dim).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # PPO parameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Memory for experience collection
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.action_probs = []
        self.dones = []
        
        # Statistics
        self.stats = {"cheap_model_uses": 0, "expensive_model_uses": 0, "rewards": []}
    
    def state_to_tensor(self, state):
        """Convert the state tuple to a tensor representation"""
        # Convert categorical variables to one-hot encoding
        complexity_map = {"simple": 0, "moderate": 1, "complex": 2}
        length_map = {"short": 0, "medium": 1, "long": 2}
        
        complexity, length = state
        
        # One-hot encode complexity (3 values)
        complexity_one_hot = [0, 0, 0]
        complexity_one_hot[complexity_map[complexity]] = 1
        
        # One-hot encode length (3 values)
        length_one_hot = [0, 0, 0]
        length_one_hot[length_map[length]] = 1
        
        # Combine into one vector
        state_vector = complexity_one_hot + length_one_hot
        
        return torch.FloatTensor(state_vector).to(self.device)
    
    def choose_model(self, state):
        """Choose a model based on the current policy"""
        state_tensor = self.state_to_tensor(state)
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
        
        # Convert to numpy for sampling
        action_probs_np = action_probs.cpu().numpy()
        
        # Sample action from the probability distribution
        action = np.random.choice(self.num_models, p=action_probs_np)
        
        # Track usage statistics
        if action == 0:
            self.stats["cheap_model_uses"] += 1
        else:
            self.stats["expensive_model_uses"] += 1
        
        # Store the probability of the selected action
        action_prob = action_probs[action].item()
        
        return action, self.models[action], action_prob

    def remember(self, state, action, reward, next_state, action_prob, done=False):
        """Store experience in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.action_probs.append(action_prob)
        self.dones.append(done)
    
    def compute_returns(self):
        """Compute returns and advantages for all stored rewards"""
        returns = []
        advantages = []
        
        # Convert states to tensors for value estimation
        states_tensor = torch.stack([self.state_to_tensor(s) for s in self.states])
        next_states_tensor = torch.stack([self.state_to_tensor(s) for s in self.next_states])
        
        with torch.no_grad():
            values = self.value_network(states_tensor).squeeze()
            next_values = self.value_network(next_states_tensor).squeeze()
        
        # Convert to numpy
        values = values.cpu().numpy()
        next_values = next_values.cpu().numpy()
        
        # Calculate returns and advantages
        for i in reversed(range(len(self.rewards))):
            # If this is the last step or if the episode is done
            if i == len(self.rewards) - 1 or self.dones[i]:
                next_return = 0
            else:
                next_return = returns[0]
            
            # Calculate return (discounted reward)
            current_return = self.rewards[i] + self.gamma * next_return
            returns.insert(0, current_return)
            
            # Calculate advantage
            if self.dones[i]:
                advantage = current_return - values[i]
            else:
                advantage = self.rewards[i] + self.gamma * next_values[i] - values[i]
            
            advantages.insert(0, advantage)
        
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)

    def update_policy(self):
        """Update policy and value networks using PPO"""
        # If there are no experiences to learn from, return
        if len(self.states) == 0:
            return
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns()
        
        # Convert states and actions to tensors
        states_tensor = torch.stack([self.state_to_tensor(s) for s in self.states])
        actions = torch.LongTensor(self.actions).to(self.device)
        old_action_probs = torch.FloatTensor(self.action_probs).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(self.ppo_epochs):
            # Create random indices
            indices = torch.randperm(len(self.states))
            
            # Create mini-batches
            for start_idx in range(0, len(self.states), self.batch_size):
                # Get mini-batch indices
                idx = indices[start_idx:start_idx + self.batch_size]
                
                # Get mini-batch data
                mb_states = states_tensor[idx]
                mb_actions = actions[idx]
                mb_old_action_probs = old_action_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                
                # Forward pass
                action_probs = self.policy_network(mb_states)
                values = self.value_network(mb_states).squeeze()
                
                # Get probabilities of the actions we actually took
                actions_one_hot = F.one_hot(mb_actions, num_classes=self.num_models).float()
                current_action_probs = torch.sum(action_probs * actions_one_hot, dim=1)
                
                # Compute ratio
                ratio = current_action_probs / mb_old_action_probs
                
                # Compute surrogate losses
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                
                # Compute policy loss, value loss, and entropy
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean()
                
                # Compute total loss (policy + value + entropy bonus)
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Backward pass and optimize
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()
        
        # Clear memory after updating
        self.clear_memory()
    
    def clear_memory(self):
        """Clear stored experiences"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.action_probs = []
        self.dones = []

    def calculate_reward(self, model_index, is_correct):
        """Calculate reward based on model performance and cost"""
        model_cost = self.models[model_index]['cost']
        # Base reward: 1 for correct, 0 for incorrect
        performance_score = 1 if is_correct else 0
        # Higher cost penalty for incorrect answers
        cost_factor = 0.05 if is_correct else 0.1
        reward = performance_score - (cost_factor * model_cost)
        self.stats["rewards"].append(reward)
        return reward
    
    def get_state(self, text):
        """Analyze text to determine state features"""
        # Length discretization
        num_words = len(text.split())
        if num_words < 50:
            length = "short"
        elif num_words < 200:
            length = "medium"
        else:
            length = "long"
        
        # Complexity discretization using readability
        complexity_score = self.calculate_complexity(text)
        if complexity_score < 30:
            complexity = "simple"
        elif complexity_score < 70:
            complexity = "moderate" 
        else:
            complexity = "complex"
        
        return (complexity, length)
    
    def calculate_complexity(self, text):
        """Calculate text complexity using various metrics"""
        # Option 1: Readability metrics
        readability_score = textstat.flesch_kincaid_grade(text)
        
        # Option 2: Vocabulary diversity
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Option 3: Sentence complexity
        sentences = nltk.sent_tokenize(text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Combined score
        return readability_score + (lexical_diversity * 50) + (avg_sentence_length * 2)
    
    def save_model(self, policy_path, value_path):
        """Save the policy and value networks"""
        torch.save(self.policy_network.state_dict(), policy_path)
        torch.save(self.value_network.state_dict(), value_path)
    
    def load_model(self, policy_path, value_path):
        """Load the policy and value networks"""
        self.policy_network.load_state_dict(torch.load(policy_path))
        self.value_network.load_state_dict(torch.load(value_path))
