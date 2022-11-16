# Needs to be run from conda environment with procgen and python 3.7.3

from procgen import ProcgenGym3Env
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, action_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 14 * 14, 100) 
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, action_dim)
        self.dropout = nn.Dropout(0.25)
        self.batch_norm_conv1 = nn.BatchNorm2d(16)
        self.batch_norm_conv2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.batch_norm_conv1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.batch_norm_conv2(x)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_policy(self, obs):
        logits = self.forward(obs)
        return Categorical(logits=logits)

    def get_prob(self, obs):
        return nn.functional.softmax(self.forward(obs), dim=0)

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

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

def train_PPO(env_name, lr, epochs, k, batch_size, epsilon, gamma, l):
    avg_reward_per_epoch = []

    # Make environment, policy model, value model, and optimizer
    env = ProcgenGym3Env(num=1, env_name=env_name, distribution_mode="easy")
    n_acts = 15 # Procgen action space is Discrete(15)
    policy_net = Net(n_acts)
    value_net = Net(1)
    optimizer = Adam(params=policy_net.parameters(), lr=lr)
    value_optimizer = Adam(params=value_net.parameters(), lr=lr)
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
        batch_rew = []

        
        # Run environment
        step = 0

        while True:
            rew, ob, first = env.observe()
            obs = (ob['rgb'] / 255).squeeze().transpose().reshape(1,3,64,64)
            batch_obs.append(obs)
            act = policy_net.get_action(torch.as_tensor(obs, dtype=torch.float32))
            env.act(np.array([act]))
            ep_rew.append(torch.from_numpy(rew))
            batch_acts.append(act)
            if first and step > 0:
                rew_over_ep.append(np.sum(ep_rew))
                ep_length.append(len(ep_rew))
                batch_rew += ep_rew
                batch_rtg += list(reward_to_go(ep_rew, gamma))
                if sum(ep_length) > batch_size:
                    break
                # Reset environment and some containers``
                ep_rew = []
                ep_value = []
                print('Episode Reward:', rew_over_ep[-1])
            step += 1
                
        
        # Calcuate p(a|s) for old policy
        act = torch.as_tensor(batch_acts, dtype=torch.int64)
        obs = torch.as_tensor(batch_obs, dtype=torch.float32).squeeze()
        old_prob = policy_net.get_prob(obs)[torch.arange(act.shape[0]), act].detach()
        batch_value = value_net(obs).squeeze()
        adv = torch.as_tensor(advantage(batch_value, batch_rew, gamma, l)).detach()
    
        weights = torch.as_tensor(batch_rtg)
        # Update Policy k times against PPO loss

        for step in range(k):
            optimizer.zero_grad(set_to_none=True)
            prob = policy_net.get_prob(obs)[torch.arange(act.shape[0]), act]
      
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
        print('reward over episode', sum(rew_over_ep))
        avg_reward_per_epoch.append(ep_length.mean())
    return avg_reward_per_epoch

if __name__ == "__main__":
    device = 'mps'
    env_name = 'coinrun'
    epochs = 1000
    ppo = np.array(train_PPO(env_name, lr=0.001, epochs=epochs, k=4, batch_size=256, epsilon=.2, gamma=.95, l = 1))
   