from environment import *
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from torch import nn, optim
import random
import matplotlib
import matplotlib.pyplot as plt
from gymnasium.wrappers import FlattenObservation

from collections import namedtuple, deque
from itertools import count, chain


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

env = PNDEnv()
n_actions = env.action_space.n
state, info = env.reset()
n_observation = len(state)

agent = Agent()
policy_net = agent.DRQN(n_observation, n_actions).to(device)
target_net = agent.DRQN(n_observation, n_actions).to(device)
# policy_net = DRQN(n_observation, n_actions).to(device)
# target_net = DRQN(n_observation, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001, amsgrad=True)
memory = agent.ReplayMemory(128)

steps_done = 0
episode_rewards = []

def plot_rewards(show_result=False):
    plt.figure(1)
    reward_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        # plt.title(f'Training..., RA: {RAALGO}, Nodes: {NUMNODES}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    # Take 10 episode averages and plot them too
    if len(reward_t) >= 10:
        means = reward_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            display.clear_output(wait=True)
    
num_episodes = 1000
        
for epoch in range(num_episodes):
    # Initialize the environment and state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    rewards = 0
    agent.
    # Select and perform an action
    action = select_action(state)
    # print(info)
    env.probenqueue(dflog)
    # print(info)
    observation, reward, terminated, truncated, info = env.step(action.item())
    # observation, reward, terminated, truncated, info = env.step_rlaqm(action.item(), dflog)
    reward = torch.tensor([reward], device=device)
    print(f"Iter: {i_episode}, Epoch: {epoch}, Action: {action.item()}, Reward: {reward.item()}")
    
    done = terminated or truncated
    if done:
        next_state = None
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the policy network)
    agent.optimize_model(policy_net, target_net, optimizer)

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*0.005 + target_net_state_dict[key]*(1-0.005)
    target_net.load_state_dict(target_net_state_dict)
    
    # print(f"Episode: {i_episode}/{num_episodes}, Epoch: {epoch}/{BEACONINTERVAL//TIMEEPOCH}, Action: {action.item()}, Reward: {reward.item()}")
    
    rewards += reward.item()
    steps_done += 1
    
    if done:
        episode_rewards.append(rewards)
        plot_rewards()
    
# print(f'Complete, RA: {RAALGO}, Nodes: {NUMNODES}')
plot_rewards(show_result=True)
plt.ioff()

# Save plot
plt.savefig('result.png')

plt.show()

# filename = f'policy_model_deepaaqm_{RAALGO}_{NUMNODES}'

# if writing == 1:
#     torch.save(policy_net, filename + '.pt')
    
# Save returns for each episode to csv file
df = pd.DataFrame(episode_rewards, columns=['reward'])
# df.to_csv(filename + '.csv')