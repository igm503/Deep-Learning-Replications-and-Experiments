from audioop import avg
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

def model(dims, activation=nn.Tanh):
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i == len(dims) - 2:
            layers.append(nn.Identity())
        else:
            layers.append(activation())
    return nn.Sequential(*layers)

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
    return (advantage - advantage.mean()) / np.std(advantage)

def train_PPO(env_name, hidden_layers, lr, epochs, k, batch_size, epsilon, gamma, l):
    avg_reward_per_epoch = []

    # Make environment, policy model, value model, and optimizer
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), 'State Space is Discrete'
    assert isinstance(env.action_space, Discrete), 'Action Space is Continuous'
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    policy_net = model([obs_dim] + hidden_layers + [n_acts])
    value_net = model([obs_dim] + hidden_layers + [1])
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

        
        # Run environment
        obs = env.reset()[0]
        done = False
        while True:
            batch_obs.append(obs)
            ep_value.append(value_net(torch.Tensor(obs)))
            print(torch.Tensor(obs).dtype)
            act = get_action(policy_net, torch.Tensor(obs))
            obs, rew, done, _, _ = env.step(act)
            ep_rew.append(rew)
            batch_acts.append(act)
            if len(ep_rew) > 10000:
                if len(ep_rew) % 1000 == 0:
                    print(len(ep_rew))
            if done:
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
                obs = env.reset()[0]
                done = False
        
        # Calcuate p(a|s) for old policy
        act = torch.as_tensor(batch_acts, dtype=torch.int64)
        obs = torch.as_tensor(batch_obs)
        old_prob = get_prob(policy_net, obs)[torch.arange(act.shape[0]), act].detach()
        adv = torch.as_tensor(batch_adv).detach()
    
        weights = torch.as_tensor(batch_rtg)
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
        print('reward over episode', sum(rew_over_ep))
        avg_reward_per_epoch.append(ep_length.mean())
    return avg_reward_per_epoch

if __name__ == "__main__":
    epochs = 10
    ppo = np.array(train_PPO('CartPole-v1', [30, 30], lr=.01, epochs=epochs, k=10, batch_size=1000, epsilon=.4, gamma=1, l = 1))
    ppo1 = np.array(train_PPO('CartPole-v1', [30, 30], lr=.01, epochs=epochs, k=10, batch_size=1000, epsilon=.3, gamma=1, l = 1))
    ppo2 = np.array(train_PPO('CartPole-v1', [30, 30], lr=.01, epochs=epochs, k=10, batch_size=1000, epsilon=.1, gamma=1, l = 1))
    ppo3 = np.array(train_PPO('CartPole-v1', [30, 30], lr=.01, epochs=epochs, k=10, batch_size=1000, epsilon=.2, gamma=1, l = 1))
    for i in range(3):
        ppo += np.array(train_PPO('CartPole-v1', [30, 30], lr=.01, epochs=epochs, k=10, batch_size=1000, epsilon=.4, gamma=1, l = 1))
        ppo1 += np.array(train_PPO('CartPole-v1', [30, 30], lr=.01, epochs=epochs, k=10, batch_size=1000, epsilon=.3, gamma=1, l = 1))
        ppo2 += np.array(train_PPO('CartPole-v1', [30, 30], lr=.01, epochs=epochs, k=10, batch_size=1000, epsilon=.1, gamma=1, l = 1))
        ppo3 += np.array(train_PPO('CartPole-v1', [30, 30], lr=.01, epochs=epochs, k=10, batch_size=1000, epsilon=.2, gamma=1, l = 1))

    x = np.arange(1, epochs + 1)
    plt.plot(x, ppo/4, label='ppo')
    plt.plot(x, ppo1/4, label='ppo')
    plt.plot(x, ppo2/4, label='ppo')
    plt.plot(x, ppo3/4, label='ppo')
    plt.legend()
    plt.show()