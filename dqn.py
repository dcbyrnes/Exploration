import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
import gym
import gym_maze
import sys
import time

class ReplayMemory:
    # capacity is approximately 100MB
    # def __init__(self, size=27000, shape=(10, 10, 9)):
    def __init__(self, size=5000, shape=(5, 5, 8)):
        self.size  = size
        self.reward = np.zeros(size)
        self.first_state = np.zeros((size,) + shape)
        self.second_state = np.zeros((size,) + shape)
        self.action = np.zeros(size)
        self.position = 0
        self.full = False

    def push(self, first_state, action, reward, second_state):
        self.first_state[self.position, :,:,:] = first_state
        self.second_state[self.position, :,:,:] = second_state
        # self.first_state[self.position, :,:] = first_state
        # self.second_state[self.position, :,:] = second_state
        self.action[self.position] = action
        self.reward[self.position] = reward
        self.position = (self.position + 1) % self.size
        if self.position == self.size - 1:
            self.full = True

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self), batch_size)
        return self.first_state[ind], self.action[ind], \
            self.reward[ind], self.second_state[ind]

    def last(self):
        return self.first_state[self.position-1], self.action[self.position-1], \
            self.reward[self.position-1], self.second_state[self.position-1]

    def __len__(self):
        if self.full:
            return self.size
        return self.position


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.lin = nn.Linear(200, 4)

    def forward(self, x):
        # x = self.lin(x.view(x.size(0), -1))
        # return x.view(x.size(0), 4)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.lin(x.view(x.size(0), 200))
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.conv4(x)
        return x.view(x.size(0), -1)

def select_action(model, state):
    # start with e-greedy
    epsilon = 0.1
    if np.random.uniform() > epsilon:
        return model(state).data.max(1)[1].numpy()[0]
    else:
        return np.random.randint(4)

np.random.seed(10)
envs = [gym.make('small-maze-{}-v0'.format(i)) for i in range(10)]

dqn = DQN().train()
optimizer = Adam(dqn.parameters())
batch_size = 100
gamma = 0.99
memory = ReplayMemory()

episode_steps = []

num_episodes = 100
for i_episode in range(num_episodes):
    state = env.reset()
    t = 0
    done = False
    total_reward = 0

    while not done and t < 2000:
        # acting
        # state_variable = Variable(torch.from_numpy(state[np.newaxis,:,:].astype('float32')))

        state_variable = Variable(torch.from_numpy(state[np.newaxis,:,:,:].astype('float32')))
        action = select_action(dqn, state_variable)
        next_state, reward, done, _ = env.step(action)


        total_reward += reward * gamma ** t
        memory.push(state, action, reward, next_state)
        state = np.copy(next_state)

        if i_episode > 50:
            env.render()
        # training
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(batch_size)

        action_batch = action_batch.reshape(batch_size,1)
        state_batch = Variable(torch.from_numpy(state_batch.astype('float32')))
        action_batch = Variable(torch.from_numpy(action_batch.astype('int64')))
        reward_batch = Variable(torch.from_numpy(reward_batch.astype('float32')))
        next_state_batch = Variable(torch.from_numpy(next_state_batch.astype('float32')))
        next_state_values = dqn(next_state_batch).max(1)[0]
        state_action_values = dqn(state_batch).gather(1, action_batch)

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t += 1
    episode_steps.append(t)

    print('done', t, 'steps', total_reward, 'reward')

import matplotlib.pyplot as plt

plt.plot(episode_steps)
plt.savefig('/Users/user/Desktop/foo.png')
# dqn = DQN()
# dqn(Variable(torch.from_numpy(np.zeros((1,10,10,9), dtype='float32'))))
