# Full connected의 경우 특정 노드가 전송을 독점하는 듯함.
# AGECOEFF가 너무 작은가?
# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1
# https://ropiens.tistory.com/80
# % tensorboard --logdir=runs

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from environment import *


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Set parameters
AGECOEFF = 0.5
gamma = 0.98
BUFFER_LIMIT = 50000
learning_rate = 0.0002
total_episodes = 500


# DRQN param
random_update = True    # If you want to do random update instead of sequential update
lookup_step = 20        # If you want to do random update instead of sequential update

# Number of envs param
n_nodes = 10
n_agents = 10
density = 1
max_steps = 300
model = None

# Set gym environment
env_params = {
    "n": n_nodes,
    "density": density,
    "max_episode_length": total_episodes,
    "model": model,
    "age_coeff": AGECOEFF,
    }
if model == None:
    env_params_str = f"n{n_nodes}_density{density}_age_coeff{AGECOEFF}_max_episode_length{total_episodes}"
else:
    env_params_str = f"n{n_nodes}_model{model}_age_coeff{AGECOEFF}_max_episode_length{total_episodes}"

env = PNDEnv(**env_params)
env.reset()

output_path = 'outputs/R2inforce_'+env_params_str
writer = SummaryWriter(filename_suffix=env_params_str)


# Create Policy functions
n_states = 2
n_actions = 2

pi_cum = [Policy(state_space=n_states, action_space=n_actions, buffer_limit=BUFFER_LIMIT).to(device) for _ in range(n_agents)]

# Set optimizer
optimizer_cum = [optim.Adam(pi_cum[i].parameters(), lr=learning_rate) for i in range(n_agents)]

df = pd.DataFrame(columns=['episode', 'time'] + [f'action_{i}' for i in range(n_agents)] + [f'age_{i}' for i in range(n_agents)])
appended_df = []

for i_epi in tqdm(range(total_episodes), desc="Episodes", position=0, leave=True):
    s, _ = env.reset()
    score = 0.0
    obs_cum = [s[np.array([x, x+n_agents])] for x in range(n_agents)]
    h_cum, c_cum = zip(*[pi_cum[i].init_hidden_state() for i in range(n_agents)])
    done = False

    for t in tqdm(range(max_steps), desc="   Steps", position=1, leave=False):
        prob_cum = [pi_cum[i](torch.from_numpy(obs_cum[i]).float().unsqueeze(0).unsqueeze(0).to(device), h_cum[i].to(device), c_cum[i].to(device))[0] for i in range(n_agents)]
        a_cum, h_cum, c_cum = zip(*[pi_cum[i].sample_action(torch.from_numpy(obs_cum[i]).float().unsqueeze(0).unsqueeze(0).to(device), h_cum[i].to(device), c_cum[i].to(device)) for i in range(n_agents)])
        a_cum = np.array(a_cum)
        s_prime, r, done, _, info = env.step(a_cum)
        done_mask = 0.0 if done else 1.0
        for i in range(n_agents):
            a = a_cum[i]
            pi_cum[i].put_data((r, prob_cum[i][0, 0, a]))
        obs_cum = [s_prime[np.array([x, x+n_agents])] for x in range(n_agents)]
        score += r

        df_currepoch = pd.DataFrame(data=[[i_epi, t, *a_cum, *env.get_current_age()]],
                                    columns=['episode', 'time'] + [f'action_{i}' for i in range(n_agents)] + [f'age_{i}' for i in range(n_agents)])
        appended_df.append(df_currepoch)

        if done:
            break

    for pi, optimizer in zip(pi_cum, optimizer_cum):
        train(pi, optimizer, gamma=gamma)

    print(f"n_episode: {i_epi}/{total_episodes}, score: {score}, deque_len: {len(pi_cum[0].data)}")
    writer.add_scalar('Rewards per episodes', score, i_epi)
    score = 0

for i in range(n_agents):
    torch.save(pi_cum[i].state_dict(), output_path + f"/R2_cum_{i}.pth")

df = pd.concat(appended_df, ignore_index=True)
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
df_last_ten_percent = df.iloc[int(total_episodes*0.9):]
df_last_ten_percent.to_csv(output_path + f"/log_{current_time}.csv", index=False)

if not os.path.exists(output_path):
    os.makedirs(output_path)
env.save_graph_with_labels(output_path)

writer.close()
env.close()