# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1



import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import environment as env


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)


adj = env.make_adjacency_matrix(10, model="dumbbell")
env.show_graph_with_labels(adj)


# Q_network

if __name__ == "__main__":

    # Env parameters
    model_name = "DRQN"
    env_name = "SALOHA"
    seed = 1
    exp_num = 'SEED'+'_'+str(seed)

    # Set gym environment
    # env = gym.make("CartPole-v1")
    cusenv = env.PNDEnv()
    # env = FlattenObservation(env)

    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    # seed_torch(seed)
    # env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/'+env_name+"_"+model_name+"_"+exp_num)
    writer = SummaryWriter(comment="_"+env_name+"_"+model_name+"_"+exp_num)

    # Set parameters
    batch_size = 8
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_epi_num = 16 # Start moment to train the Q network
    episodes = 650
    print_per_iter = 20
    target_update_period = 4
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 2000

    # DRQN param
    random_update = True# If you want to do random update instead of sequential update
    lookup_step = 10 # If you want to do random update instead of sequential update
    max_epi_len = 128
    max_epi_step = max_step

    

    # Create Q functions
    Q = Q_net(state_space=5, # env.observation_space.shape[0]
              action_space=cusenv.action_space.n).to(device)
    Q_target = Q_net(state_space=5, 
                     action_space=cusenv.action_space.n).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start
    
    episode_memory = EpisodeMemory(random_update=random_update, 
                                   max_epi_num=100, max_epi_len=600, 
                                   batch_size=batch_size, 
                                   lookup_step=lookup_step)

    # Train
    for i in range(episodes):
        s, _ = cusenv.reset(seed=seed)
        obs = s
        done = False
        
        episode_record = EpisodeBuffer()
        h, c = Q.init_hidden_state(batch_size=batch_size, training=False)

        for t in range(max_step):

            # Get action
            a, h, c = Q.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                      h.to(device), 
                                      c.to(device),
                                      epsilon)

            # Do action
            s_prime, r, done, _, _ = cusenv.step(a)
            obs_prime = s_prime

            # make data
            done_mask = 0.0 if done else 1.0

            episode_record.put([obs, a, r/100.0, obs_prime, done_mask])

            obs = obs_prime
            
            score += r
            score_sum += r

            if len(episode_memory) >= min_epi_num:
                train(Q, Q_target, episode_memory, device, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate)

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): # <- soft update
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break
        
        episode_memory.put(episode_record)
        
        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        if i % print_per_iter == 0 and i!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            i, score_sum/print_per_iter, len(episode_memory), epsilon*100))
            score_sum=0.0
            save_model(Q, model_name+"_"+exp_num+'.pth')

        # Log the reward
        writer.add_scalar('Rewards per episodes', score, i)
        score = 0
        
    writer.close()
    cusenv.close()