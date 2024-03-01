from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
import matplotlib.pyplot as plt

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  

#We will start from the DQN of the course
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = int(0) # index of the next cell to be filled


    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = int((self.index + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)), list(zip(*batch))))
    
    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self, config=None, model=None):


    
        if config == None:
            config = {'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'buffer_size': 1e6,
            'epsilon_min': 0.01,
            'epsilon_max': 1.,
            'epsilon_decay_period': 5000,
            'epsilon_delay_decay': 50,
            'batch_size': 200,
            'gradient_steps': 2,
            'update_target_strategy': 'ema',
            'update_target_freq': 50,
            'update_target_tau': 0.005,
            'criterion': torch.nn.SmoothL1Loss()}
        
        self.nb_neurons=256
        self.nb_actions = config['nb_actions']
        self.state_dim = env.observation_space.shape[0]

        if model == None:
            #I see that the use of a bigger model helps achieve way better results
            model = torch.nn.Sequential(nn.Linear(self.state_dim, self.nb_neurons),
                nn.ReLU(),
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(),
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(),
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(),
                nn.Linear(self.nb_neurons, self.nb_neurons), 
                nn.ReLU(),
                nn.Linear(self.nb_neurons, self.nb_neurons),
                nn.ReLU(),
                nn.Linear(self.nb_neurons, self.nb_actions))


        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

        self.path_model = 'src/{model_name}.pth'

        
        nb_actions = env.action_space.n 
        self.best_model = None
        self.best_value = 0
    
    
        
    def save(self, model_filename='model_max.pth'):
        save_dir = '/Users/sacha/Documents/GitHub/rl-class-assignment-Sacha-Elkoubi/src'
        save_path = os.path.join(save_dir, model_filename)
        torch.save(self.model.state_dict(), save_path)

    def load(self, model_filename='model_max.pth'):
        load_dir = '/Users/sacha/Documents/GitHub/rl-class-assignment-Sacha-Elkoubi/src'
        load_path = os.path.join(load_dir, model_filename)
        self.model.load_state_dict(torch.load(load_path))
        self.target_model.load_state_dict(torch.load(load_path))


    def act(self, observation, use_random=False):
      if use_random :
        return env.action_space.sample()
      else :
        with torch.no_grad():
          Q = self.model(torch.Tensor(observation).unsqueeze(0))
        return torch.argmax(Q).item()
    

    def gradient_step(self):
        #as in course
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_score=0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = self.act(state, use_random=True)
            else:
                action = self.act(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                if episode_cum_reward > best_score:  # Check if current episode return is the best
                    best_score = episode_cum_reward  # Update best score
                    torch.save(self.model.state_dict(), 'model_max.pth')  # Save best model
                    print("New best score! Model saved.")
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return



#Declare network
state_dim = env.observation_space.shape[0]
n_action= env.action_space
nb_neurons=256

# DQN config
config_test = {'nb_actions': env.action_space.n ,
        'learning_rate': 0.001,
        'gamma': 1, #finite time 
        'buffer_size': 1e6,
        'epsilon_min': 0.01,
        'epsilon_max': 1.,
        'epsilon_decay_period': 20000,
        'epsilon_delay_decay': 200,
        'batch_size': 200,
        'gradient_steps': 3,
        'update_target_strategy': 'replace',
        'update_target_freq': 400,
        'update_target_tau': 0.005,
        'criterion': torch.nn.SmoothL1Loss()}

# # Train agent
# agent = ProjectAgent(config=config_test)
# scores = agent.train(env, 250)

