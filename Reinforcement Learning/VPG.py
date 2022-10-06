import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from tqdm import tqdm
import matplotlib.pyplot as plt

def model(dims, activation=nn.ReLU):
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

def train(epochs, env_name='LunarLander-v2', hidden_layers=[32], lr=1e-2, 
          batch_size=50, val_baseline=True):

    avg_reward_per_epoch = []
    # Make environment, model, and optimizer
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), 'State Space is Discrete'
    assert isinstance(env.action_space, Discrete), 'Action Space is Continuous'
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    policy_net = model([obs_dim] + hidden_layers + [n_acts])
    optimizer = Adam(params=policy_net.parameters(), lr=lr)
    for epoch in range(epochs):
        # Make Data Containers
        batch_obs = []
        batch_rtg = []
        batch_acts = []
        ep_rew = []
        rew_per_ep = []
        ep_length = []
        # Run environment
        obs = env.reset()[0]
        done = False
        while True:
            batch_obs.append(obs)
            act = get_action(policy_net, torch.Tensor(obs))
            obs, rew, done, _, _ = env.step(act)
            ep_rew.append(rew)
            batch_acts.append(act)
            if done:
                rew_per_ep.append(np.sum(ep_rew))
                ep_length.append(len(ep_rew))
                batch_rtg += list(reward_to_go(ep_rew, gamma))
                if sum(ep_length) > batch_size:
                    break
                # Reset environment and some containers
                ep_rew = []
                obs = env.reset()[0]
                done = False

        # take a single policy gradient update step
        optimizer.zero_grad()

        # adjust weights by value function baseline
        returns = torch.as_tensor(np.array(batch_weights), dtype=torch.float32)
        if val_baseline:
            baseline = torch.as_tensor(batch_vals, dtype=torch.float32).clone().detach()
            weights = returns - baseline
        else:
            weights = returns
        batch_loss = compute_loss(obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
                                  act=torch.as_tensor(np.array(batch_acts), dtype=torch.int32),
                                  weights=weights
                                  )
        batch_loss.backward()
        optimizer.step()

        # take a several value function update step
        if val_baseline:
            val_batch_size = 10
            epoch_length = len(batch_weights)
            num_batches = epoch_length // val_batch_size
            obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
            for i in range(num_batches):
                value_optimizer.zero_grad()
                val_batch_loss = compute_value_loss(obs=obs_tensor[10*i:10*i + 10],
                                                    weights=returns[10*i:10*i + 10])
                val_batch_loss.backward()
                value_optimizer.step()
        if val_baseline:
            return batch_loss, batch_rets, batch_lens, val_batch_loss
        else:
            return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        if val_baseline:
            batch_loss, batch_rets, batch_lens, val_batch_loss = train_one_epoch()
        else:
            batch_loss, batch_rets, batch_lens = train_one_epoch()
        len_over_training.append(np.mean(batch_lens))
        rew_over_training.append(np.mean(batch_rets))

        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    return torch.Tensor(rew_over_training).unsqueeze(dim=0)

if __name__ == '__main__':
    epochs=10
    sample_size = 10
    # Comparing VPG with value function baseline to VPG with no baseline
    baseline = train(env_name='LunarLander-v2', lr=.01, val_baseline=True, epochs=epochs)
    no_baseline = train(env_name='LunarLander-v2', lr=.01, val_baseline=False, epochs=epochs)
    for i in tqdm(range(sample_size-1)):
        baseline = torch.cat([baseline, train(env_name='LunarLander-v2', lr=.01, val_baseline=True, epochs=epochs)])
        no_baseline = torch.cat([no_baseline, train(env_name='LunarLander-v2', lr=.01, val_baseline=False, epochs=epochs)])

    baseline_mean = baseline.mean(dim=0)
    no_baseline_mean = no_baseline.mean(dim=0)
    baseline_error = baseline.std(dim=0)
    no_baseline_error = no_baseline.std(dim=0)
    x = np.arange(1, epochs+1)
    plt.errorbar(x=x, y=baseline_mean, yerr=baseline_error, label='val BL')
    plt.errorbar(x=x, y=no_baseline_mean, yerr=no_baseline_error, label='no BL')
    plt.legend()
    plt.show()