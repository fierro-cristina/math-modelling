import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import math
import random
from collections import deque

import matplotlib.pyplot as plt


Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)

class replayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


initial_epsilon = 1.0
epsilon_decay = 500
min_epsilon = 0.01

epsilon_per_episode = lambda episode_idx: min_epsilon + (initial_epsilon - min_epsilon) * math.exp(-1. * episode_idx / epsilon_decay)


class DQNmodel(nn.Module):
    def __init__(self, n_inputs, n_episodes, n_layers, n_neurons):
        super(DQNmodel, self).__init__()
        if n_layers == 0:
            self.layers = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons),
                                        nn.ReLU(),
                                        nn.Linear(n_neurons, env.action_space.n))
        if n_layers == 1:
            self.layers = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons),
                                        nn.ReLU(),
                                        nn.Linear(n_neurons, n_neurons),
                                        nn.ReLU(),
                                        nn.Linear(n_neurons, env.action_space.n))
        if n_layers == 2:
            self.layers = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons),
                                        nn.ReLU(),
                                        nn.Linear(n_neurons, n_neurons),
                                        nn.ReLU(),
                                        nn.Linear(n_neurons, n_neurons),
                                        nn.ReLU(),
                                        nn.Linear(n_neurons, env.action_space.n))
    def forward(self, t):
        return self.layers(t)

    def takeAction(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action

env_id = "CartPole-v0"
env = gym.make(env_id)

model = DQNmodel(env.observation_space.shape[0], env.action_space.n, 2, 128)

optimizer = optim.Adam(model.parameters())

replay_mem_buf = replayMemory(1000)

def loss_calc(batch_size):
    state, action, reward, next_state, done = replay_mem_buf.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_vals = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_val = next_q_vals.max(1)[0]
    expected_q_val = reward + gamma * next_q_val * (1 - done)

    loss = (q_value - Variable(expected_q_val.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def plot(episode_idx, rewards, losses):
    plt.title('Avg reward %s. in %s. episodes' % (episode_idx, np.mean(rewards[-10:])))
    plt.plot(rewards, color = 'blue')
    plt.xlabel('episodes'), plt.ylabel('reward score')
    plt.grid(True)
    plt.show()
    plt.title('Loss in {} episodes'.format(episode_idx))
    plt.plot(losses, color = 'green')
    plt.xlabel('episodes'), plt.ylabel('loss score')
    plt.grid(True)
    plt.show()


n_episodes = 1000
batch_size = 32
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0

reward_plot = []
losses_plot = []

state = env.reset()

for episode_idx in range(1, n_episodes +1):
    epsilon = epsilon_per_episode(episode_idx)
    action = model.takeAction(state, epsilon)

    next_state, reward, done, _ = env.step(action)

    replay_mem_buf.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(replay_mem_buf) > batch_size:
        loss = loss_calc(batch_size)
        losses.append(loss.item())

    if episode_idx % 200 == 0:
        reward_plot.append(all_rewards)
        losses_plot.append(all_rewards)

plot(episode_idx, all_rewards, losses)
