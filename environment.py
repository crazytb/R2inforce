from collections import deque
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Box, MultiBinary
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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
            self.max_steps = kwargs.get("max_steps", 300)
            self.age_coeff = kwargs.get("age_coeff", 0.1)

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
        self.remain_steps = self.max_steps

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

        self._current_age += 1/self.max_steps
        self._current_age = np.clip(self._current_age, 0, 1)
        self._current_age[idx_success] = 0
        self.remain_steps -= 1

        # reward = n_txtrial/self.max_episode_length - max(self._current_age) # 보낸 갯수만큼 보상을 준다.
        reward = n_txtrial - self.age_coeff*max(self._current_age) # 보낸 갯수만큼 보상을 준다.

        done = (self.remain_steps == 0)
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


class Policy(nn.Module):
    def __init__(self, state_space=2, action_space=2, buffer_limit=10000):
        super(Policy, self).__init__()
        self.data = []
        self.state_space = state_space
        self.hidden_space = 4
        self.action_space = action_space
        self.linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space)
        self.linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.linear1(x))
        x, (new_h, new_c) = self.lstm(x, (h, c))
        x = F.softmax(self.linear2(x), dim=2)
        return x, new_h, new_c

    def put_data(self, transition):
        self.data.append(transition)

    def sample_action(self, obs, h, c):
        output = self.forward(obs, h, c)
        # Select action with respect to the action probabilities
        action = torch.squeeze(output[0]).multinomial(num_samples=1)
        return action.item(), output[1], output[2]

    def init_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_space, device=device), torch.zeros(1, 1, self.hidden_space, device=device)

def train(pi, optimizer, gamma=0.98):
    R = 0
    policy_loss = []
    optimizer.zero_grad()
    for r, prob in reversed(pi.data):
        R = r + gamma * R
        loss = -torch.log(prob) * R # Negative score function x reward
        policy_loss.append(loss)
    sum(policy_loss).backward()
    optimizer.step()
    pi.data = []