# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
# https://ropiens.tistory.com/80
# % tensorboard --logdir=runs

import sys
import os
from typing import Dict, List, Tuple
import gymnasium as gym
import collections
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from environment import *


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Set parameters
batch_size = 8
learning_rate = 1e-3
buffer_len = int(100000)
min_epi_num = 20 # Start moment to train the Q network
episodes = 650
target_update_period = 4
eps_start = 0.1
eps_end = 0.001
eps_decay = 0.995
tau = 1e-2

# DRQN param
random_update = True    # If you want to do random update instead of sequential update
lookup_step = 20        # If you want to do random update instead of sequential update
max_epi_len = 100 

# Number of envs param
n_nodes = 10
density = 0.5
max_step = 1000
model = "dumbbell"

# Set gym environment
env_params = {
    "n": n_nodes,
    "density": density,
    "max_episode_length": max_step,
    "model": model
    }
env_params_str = f"n{n_nodes}_density{density}_max_episode_length{max_step}_model{model}"

# Make env and reset it
env = PNDEnv(**env_params)
env.reset()

# Set the seed
seed = 42
np.random.seed(seed)
random.seed(seed)
seed_torch(seed)

# default `log_dir` is "runs" - we'll be more specific here
run_path = 'runs/DQN_'+env_params_str
output_path = 'outputs/DQN_'+env_params_str
writer = SummaryWriter(run_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
env.save_graph_with_labels(output_path)


# Create Q functions
Q_cum = [DQN(state_space=2, action_space=2).to(device) for _ in range(n_nodes)]
Q_target_cum = [DQN(state_space=2, action_space=2).to(device) for _ in range(n_nodes)]
[Q_target_cum[i].load_state_dict(Q_cum[i].state_dict()) for i in range(n_nodes)]
# Q = DRQN(state_space=2, 
#           action_space=2).to(device)
# Q_target = DRQN(state_space=2, 
#                  action_space=2).to(device)
# Q_target.load_state_dict(Q.state_dict())

# Set optimizer
score = 0
score_sum = 0
optimizer_cum = [optim.Adam(Q_cum[i].parameters(), lr=learning_rate) for i in range(n_nodes)]
# optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

epsilon = eps_start

episode_memory = [EpisodeMemory(random_update=random_update, max_epi_num=100, max_epi_len=600, batch_size=batch_size, lookup_step=lookup_step) for _ in range(n_nodes)]
# EpisodeMemory(random_update=random_update, max_epi_num=100, max_epi_len=600, batch_size=batch_size, lookup_step=lookup_step)

# Train
for i_epi in range(episodes):
    s, _ = env.reset()
    obs_cum = [s[np.array([x, x+n_nodes])] for x in range(n_nodes)]
    # obs = s[np.array([0, n_nodes])] # Use only Position of Cart and Pole, # "-2" should be considered.
    done = False
    
    episode_record_cum = [EpisodeBuffer() for _ in range(n_nodes)]
    # episode_record = EpisodeBuffer()
    # h_cum, c_cum = zip(*[Q_cum[i].init_hidden_state(batch_size=batch_size, training=False) for i in range(n_nodes)])
    # h, c = Q.init_hidden_state(batch_size=batch_size, training=False)

    for t in range(max_step):
        # Get action
        a_cum = [Q_cum[i].sample_action(torch.from_numpy(obs_cum[i]).float().to(device).unsqueeze(0).unsqueeze(0), epsilon) for i in range(n_nodes)]
        a = np.array(a_cum)
        # Do action
        s_prime, r, done, _, _ = env.step(a)
        # obs_prime = s_prime # "-2" should be considered.
        obs_prime_cum = [s_prime[np.array([x, x+n_nodes])] for x in range(n_nodes)]

        # make data
        done_mask = 0.0 if done else 1.0

        for i_n in range(n_nodes):
            episode_record_cum[i_n].put([obs_cum[i_n], a[i_n], r/100.0, obs_prime_cum[i_n], done_mask])

        obs_cum = obs_prime_cum
        
        score = r
        score_sum += r

        for i_m in range(n_nodes):
            if len(episode_memory[i_m]) >= min_epi_num:
                train(Q_cum[i_m], Q_target_cum[i_m], episode_memory[i_m], device, optimizer=optimizer_cum[i_m], batch_size=batch_size, learning_rate=learning_rate)

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- naive update
                    for target_param, local_param in zip(Q_target_cum[i_m].parameters(), Q_cum[i_m].parameters()): # <- soft update
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
            
        if done:
            break
    
    for i_j in range(n_nodes):
        episode_memory[i_j].put(episode_record_cum[i_j])
    
    epsilon = max(eps_end, epsilon * eps_decay) # Linear annealing
    
    print(f"n_episode: {i_epi}, score: {score_sum}, n_buffer: {len(episode_memory)}, eps: {epsilon*100}%")
        
    # Log the reward
    writer.add_scalar('Rewards per episodes', score_sum, i_epi)
    score = 0
    score_sum = 0.0
    
for i in range(n_nodes):
    model_name = output_path + "_" + str(i)
    torch.save(Q_cum[i].state_dict(), model_name+".pth")
    
writer.close()
env.close()