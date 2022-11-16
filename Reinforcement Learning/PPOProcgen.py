# Needs to be run from conda environment with procgen and python 3.7.3

from os import device_encoding
from procgen import ProcgenGym3Env
import gym3
from gym3 import types_np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, n_obs, n_out, hiddens):
        super().__init__()
        self.layers = []
        self.net = nn.Sequential(nn.Linear(n_obs, hiddens[0]))
        for i in range(len(hiddens) - 2):
            self.net.append(nn.Tanh())
            self.net.append(nn.Linear(hiddens[i], hiddens[i+1]))
        self.net.append(nn.Tanh())
        self.net.append(nn.Linear(hiddens[i + 1], n_out))
    def forward(self, x):
        return self.net(x)

def get_policy(model, obs):
    logits = model(obs)
    return Categorical(logits=logits)

def get_prob(model, obs):
    return nn.functional.softmax(model(obs))

def get_action(model, obs):
    return get_policy(model, obs).sample().item()

def reward_to_go(reward_list, gamma):
    reward_to_go = torch.zeros((len(reward_list)))
    for i in reversed(range(len(reward_list)-1)):
        if i == len(reward_list) - 1:
            reward_to_go[i] = reward_list[i]
        else:
            reward_to_go[i] = reward_list[i] + reward_to_go[i+1]*gamma
    return reward_to_go

def advantage(value, rew, gamma, l):
    advantage = np.zeros_like(rew)
    for t in reversed(range(len(rew))):
        if t == len(rew) - 1:
            advantage[t] = rew[t] - value[t]
        else:
            delta_t = rew[t] + gamma * value[t + 1] - value[t]
            advantage[t] = delta_t + (gamma * l) * advantage[t + 1]
    return (advantage - advantage.mean()) / (np.std(advantage) + 1e-6)

def train_PPO(env_name, hidden_layers, lr, epochs, k, batch_size, epsilon, gamma, l):
    avg_reward_per_epoch = []

    # Make environment, policy model, value model, and optimizer
    env = ProcgenGym3Env(num=1, env_name=env_name, distribution_mode="easy")

    obs_dim = 64 * 64 * 3 # Procgen obs are 64 x 64 rgb screens
    n_acts = 15 # Procgen action space is Discrete(15)
    policy_net = Model(obs_dim, n_acts, hidden_layers).to(device)
    value_net = Model(obs_dim, 1, hidden_layers).to(device)
    optimizer = Adam(params=policy_net.parameters(), lr=lr)
    value_optimizer = Adam(params=value_net.parameters(), lr=lr)
    param_count = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print('num params:', param_count)
    for epoch in range(epochs):
        
        # Make Data Containers
        batch_obs = []
        batch_rtg = []
        batch_acts = []
        batch_value = []
        batch_adv = []
        ep_rew = []
        ep_value = []
        ep_length = []
        rew_over_ep = []

        
        # Run environment
        step = 0
        while True:
            rew, obs, first = env.observe()
            obs = obs['rgb'].reshape((obs_dim)) / 255 # converting to 1-d array of [0,1] rgb
            batch_obs.append(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device) # convert to tensor for network input
            ep_value.append(value_net(obs))
            act = get_action(policy_net, obs)
            env.act(np.array([act]))
            ep_rew.append(torch.from_numpy(rew).to(device))
            batch_acts.append(act)
            if first and step > 0:
                rew_over_ep.append(np.sum(ep_rew))
                ep_length.append(len(ep_rew))
                batch_value += ep_value
                batch_rtg += list(reward_to_go(ep_rew, gamma))
                batch_adv += list(advantage(ep_value, ep_rew, gamma, l))
                if sum(ep_length) > batch_size:
                    break
                # Reset environment and some containers``
                ep_rew = []
                ep_value = []
                # obs = env.reset()[0]
                done = False
            step += 1
            if step % 1000 == 0:
                print(step)
        
        # Calcuate p(a|s) for old policy
        act = torch.as_tensor(batch_acts, dtype=torch.int64).to(device)
        obs = torch.as_tensor(batch_obs, dtype=torch.float32).to(device)
        old_prob = get_prob(policy_net, obs)[torch.arange(act.shape[0]), act].detach().to(device)
        adv = torch.as_tensor(batch_adv).detach().to(device)
    
        weights = torch.as_tensor(batch_rtg).to(device)
       # adv = (weights - value_net(obs).squeeze()).detach()
        # Update Policy k times against PPO loss
        for step in range(k):
            optimizer.zero_grad(set_to_none=True)
            prob = get_prob(policy_net, obs)[torch.arange(act.shape[0]), act]
      
            ratio = prob/old_prob
            
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
            loss = (-torch.min(surr1, surr2)).mean()
      
            loss.backward()
            optimizer.step()
            # Update value function
            value_optimizer.zero_grad(set_to_none=True)
            value_loss = ((value_net(obs).squeeze() - weights)**2).mean()
            value_loss.backward()
            value_optimizer.step()
        # Print Run Info
        
        print('value loss', value_loss.item())
        print('epoch:', epoch)
        ep_length = np.array(ep_length)
        print('mean ep length:', ep_length.mean())
        print('total reward', sum(rew_over_ep))
        avg_reward_per_epoch.append(ep_length.mean())
    return avg_reward_per_epoch

if __name__ == "__main__":
    device = 'cpu'
    env_name = 'coinrun'
    epochs = 1000
    ppo = np.array(train_PPO(env_name, [100, 100, 30], lr=.001, epochs=epochs, k=3, batch_size=256, epsilon=.2, gamma=.95, l = .999))
   