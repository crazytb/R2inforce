import sys
import collections
from typing import Dict, List, Tuple
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box, MultiBinary
import gymnasium as gym
import math
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from numpy.random import default_rng

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  
else:
    device = torch.device("cpu")

# Parameters
POWERCOEFF = 0.1
AGECOEFF = 0.1

class PNDEnv(Env):
    def __init__(self, **kwargs):
            """
            Initialize the PNDEnv class.

            Parameters:
            - n (int): The number of nodes in the environment.
            - density (float): The density of the environment.
            - max_epi (int): The maximum number of episodes.
            - model (str): The model to be used.

            Returns:
            None
            """
            super(PNDEnv, self).__init__()
            self.n = kwargs.get("n", 10)
            self.density = kwargs.get("density", 0.5)
            self.model = kwargs.get("model", None)
            self.max_episode_length = kwargs.get("max_episode_length", 2000)

            # Actions we can take 0) transmit and 1) listen
            self.action_space = MultiBinary(self.n)
            # Observation space
            self.observation_space = spaces.Dict({
                "current_age": Box(low=0, high=1, shape=(self.n, 1)),
                "prev_result": MultiBinary([self.n, 1]),
                # 0: Listening, 1: Transmitting
            })

    def get_obs(self):
        current_age = np.reshape(self._current_age, newshape=(self.n))
        prev_result = np.reshape(self._prev_result, newshape=(self.n))
        return np.concatenate([current_age, prev_result])
    
    def get_info(self):
        print("Current Age, Prev Result")
        for i in range(self.n):
            print(f"Node {i}: {self._current_age[i]}, {self._prev_result[i]}")
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # State reset
        self._current_age = np.zeros(self.n)
        self._prev_result = np.zeros(self.n)
        self.adjacency_matrix = self.make_adjacency_matrix()  # Adjacency matrix
        self.where_packet_is_from = np.array([None]*self.n)
        self.episode_length = 300

        observation = self.get_obs() 
        return observation, None

    def step(self, action: np.array):  # 여기 해야 함.
        # Check if the action is valid. Action length must be equal to the number of nodes and action must be 0 or 1. 

        assert len(action) == len(self._prev_result), "Action length must be equal to the number of nodes."
        assert all([a in [0, 1] for a in action]), "Action must be 0 or 1."
        self.where_packet_is_from = np.array([None]*self.n)
        self._prev_result = action

        action_tiled = np.tile(action.reshape(-1, 1), (1, self.n))
        txrx_matrix = np.multiply(self.adjacency_matrix, action_tiled)

        for i in np.where(action==1)[0]:
            txrx_matrix[:, i] = 0

        collided_index = np.sum(txrx_matrix, axis=0)>1
        txrx_matrix[:, collided_index] = 0

        # n_txtrial = np.count_nonzero(action)
        idx_success = np.where(np.sum(txrx_matrix, axis=1)!=0)[0]
        n_txtrial = len(idx_success)

        self._current_age += 1/self.max_episode_length
        self._current_age = np.clip(self._current_age, 0, 1)
        self._current_age[idx_success] = 0
        self.episode_length -= 1
        
        # reward = n_txtrial/self.max_episode_length - max(self._current_age) # 보낸 갯수만큼 보상을 준다.
        reward = n_txtrial - AGECOEFF*max(self._current_age) # 보낸 갯수만큼 보상을 준다.

        done = (self.episode_length == 0)
        observation = self.get_obs()

        return observation, reward, False, done, None

    
    def render(self):
        # Implement viz
        pass
    
    def make_adjacency_matrix(self) -> np.ndarray:
        """Make adjacency matrix of a clique network.
        
        Args:
            n (int): Number of nodes.
            density (float): Density of the clique network.
        
        Returns:
            np.ndarray: Adjacency matrix.
        """
        if self.density < 0 or self.density > 1:
            raise ValueError("Density must be between 0 and 1.")
        
        n_edges = int(self.n * (self.n - 1) / 2 * self.density)
        adjacency_matrix = np.zeros((self.n, self.n))

        if self.model == "dumbbell":
            adjacency_matrix[0, self.n-1] = 1
            adjacency_matrix[self.n-1, 0] = 1
            for i in range(1, self.n//2):
                adjacency_matrix[0, i] = 1
                adjacency_matrix[i, 0] = 1
            for i in range(self.n//2+1, self.n):
                adjacency_matrix[i-1, self.n-1] = 1
                adjacency_matrix[self.n-1, i-1] = 1
        elif self.model == "linear":
            for i in range(1, self.n):
                adjacency_matrix[i-1, i] = 1
                adjacency_matrix[i, i-1] = 1
        else:
            for i in range(1, self.n):
                adjacency_matrix[i-1, i] = 1
                adjacency_matrix[i, i-1] = 1
                n_edges -= 1
            # If the density of the current adjacency matrix is over density, return it.
            if n_edges <= 0:
                return adjacency_matrix
            else:
                arr = [1]*n_edges + [0]*((self.n-1)*(self.n-2)//2 - n_edges)
                np.random.shuffle(arr)
                for i in range(0, self.n):
                    for j in range(i+2, self.n):
                        adjacency_matrix[i, j] = arr.pop()
                        adjacency_matrix[j, i] = adjacency_matrix[i, j]
        return adjacency_matrix
    
    def show_adjacency_matrix(self):
        print(self.adjacency_matrix)

    def save_graph_with_labels(self, path):
        rows, cols = np.where(self.adjacency_matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx(G, pos=pos, with_labels=True)
        plt.savefig(path + '/adj_graph.png')
        
    def get_current_age(self):
        return self._current_age
        

# Policy network
class Policy(nn.Module):
    """
    DRQN class represents the deep Q-network model for reinforcement learning.

    Args:
        state_space (int): The size of the state space.
        action_space (int): The size of the action space.

    Attributes:
        hidden_space (int): The size of the hidden space.
        state_space (int): The size of the state space.
        action_space (int): The size of the action space.
        Linear1 (nn.Linear): The first linear layer.
        lstm (nn.LSTM): The LSTM layer.
        Linear2 (nn.Linear): The second linear layer.

    Methods:
        forward(x, h, c): Performs forward pass through the network.
        sample_action(obs, h, c, epsilon): Samples an action based on the current observation and epsilon-greedy policy.
        init_hidden_state(batch_size, training): Initializes the hidden state of the LSTM.

    """

    def __init__(self, state_space=None, action_space=None):
        super(Policy, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.data = []
        
        self.hidden_space = 4
        self.state_space = state_space
        self.action_space = action_space

        self.linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        """
        Performs forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.
            h (torch.Tensor): The previous hidden state of the LSTM.
            c (torch.Tensor): The previous cell state of the LSTM.

        Returns:
            torch.Tensor: The output tensor.
            torch.Tensor: The new hidden state of the LSTM.
            torch.Tensor: The new cell state of the LSTM.

        """
        x = F.relu(self.linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c))
        x = F.softmax(self.linear2(x), dim=2)
        return x, new_h, new_c

    def put_data(self, transition):
        self.data.append(transition)
    
    def sample_action(self, obs, h, c):
        """
        Samples an action based on the current observation and epsilon-greedy policy.

        Args:
            obs (torch.Tensor): The current observation.
            h (torch.Tensor): The previous hidden state of the LSTM.
            c (torch.Tensor): The previous cell state of the LSTM.
            epsilon (float): The exploration rate.

        Returns:
            int: The sampled action.
            torch.Tensor: The new hidden state of the LSTM.
            torch.Tensor: The new cell state of the LSTM.

        """
        output = self.forward(obs, h, c)
        action = torch.squeeze(output[0]).multinomial(num_samples=1)

        return action.item(), output[1], output[2]

    def init_hidden_state(self, batch_size, training=None):
        """
        Initializes the hidden state of the LSTM.

        Args:
            batch_size (int): The batch size.
            training (bool): Indicates whether it is a training step or not.

        Returns:
            torch.Tensor: The initial hidden state of the LSTM.
            torch.Tensor: The initial cell state of the LSTM.

        Raises:
            AssertionError: If the training parameter is not specified.

        """
        assert training is not None, "training step parameter should be determined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])


class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False, 
                       max_epi_num=100, max_epi_len=500,
                       batch_size=1,
                       lookup_step=None):
        self.random_update = random_update # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit('It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update: # Random update
            sampled_episodes = random.sample(self.memory, self.batch_size)
            
            check_flag = True # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step: # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_step+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################           
        else: # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs']) # buffers, sequence_length

    def __len__(self):
        return len(self.memory)


class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step]
            action = action[idx:idx+lookup_step]
            reward = reward[idx:idx+lookup_step]
            next_obs = next_obs[idx:idx+lookup_step]
            done = done[idx:idx+lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


def train(pi=None, 
          episode_memory=None,
          device=None, 
          optimizer = None,
          batch_size=1,
          learning_rate=1e-3,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    observations = torch.FloatTensor(observations.reshape(batch_size,seq_len,-1)).to(device)
    actions = torch.LongTensor(actions.reshape(batch_size,seq_len,-1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size,seq_len,-1)).to(device)
    next_observations = torch.FloatTensor(next_observations.reshape(batch_size,seq_len,-1)).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size,seq_len,-1)).to(device)

    h, c = pi.init_hidden_state(batch_size=batch_size, training=True)
    q_out, _, _ = pi(observations, h.to(device), c.to(device))
    q_a = q_out.gather(2, actions)

    # Update Network
    R = 0
    optimizer.zero_grad()
    for r, prob in zip(reversed(rewards), reversed(q_a)):
        R = r + gamma * R
        loss = -torch.log(prob) * R
        loss.mean().backward(retain_graph=True)
    optimizer.step()

def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)