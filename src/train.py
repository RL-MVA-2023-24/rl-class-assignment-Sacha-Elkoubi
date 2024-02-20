import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gymnasium as gym
from env_hiv import HIVPatient


class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ProjectAgent:
    def __init__(self, observation_size: int=6, action_size: int=4, gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01, learning_rate: float = 0.001,
                 batch_size: int = 64, replay_memory_size: int = 10000, target_update_freq: int = 100):
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_network = QNetwork(observation_size, action_size)
        self.target_q_network = QNetwork(observation_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.replay_memory = []
        self.replay_memory_size = replay_memory_size
        self.timestep = 0

    def act(self, observation: np.ndarray) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        observation = torch.Tensor(observation).unsqueeze(0)
        q_values = self.q_network(observation)
        return torch.argmax(q_values).item()

    def remember(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool):
        self.replay_memory.append((observation, action, reward, next_observation, done))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return
        minibatch = random.sample(self.replay_memory, self.batch_size)
        observations, targets = [], []
        for observation, action, reward, next_observation, done in minibatch:
            target = reward
            if not done:
                next_observation = torch.FloatTensor(next_observation).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_q_network(next_observation).detach())
            target_f = self.q_network(torch.FloatTensor(observation).unsqueeze(0))
            target_f[0][action] = target
            observations.append(observation)
            targets.append(target_f)
        observations = torch.FloatTensor(observations)
        targets = torch.cat(targets)
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(self.q_network(observations), targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.timestep += 1
        if self.timestep % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str="./model_max.pth") -> None:
        self.q_network.load_state_dict(torch.load(path))

def train(agent: ProjectAgent, env, episodes: int):
    max_total_reward = float(0)
    for episode in range(episodes):
        observation = env.reset()
        if isinstance(observation, tuple):  # Check if observation is a tuple
            observation = observation[0]  # Take the first element of the tuple
        total_reward = 0
        done = False
        for _ in range(episodes):
            action = agent.act(observation)
            next_observation, reward, _ , _, _ = env.step(action)
            agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation
            total_reward += reward
            agent.replay()
    
        
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
        
        if  total_reward>max_total_reward:
            max_total_reward=total_reward
            agent.save("model_max.pth")
            

# Example usage:
# Initialize environment and agent
#env = HIVPatient()
#agent = ProjectAgent(observation_size=env.observation_space.shape[0], action_size=env.action_space.n)
# Train the agent
#train(agent, env, episodes=100)
