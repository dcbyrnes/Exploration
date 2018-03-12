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

class CoreNet(nn.Module):
    def __init__(self):
        super(CoreNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        return x.view(x.size(0), 200)

class HeadNet(nn.Module):
    def __init__(self, core):
        super(HeadNet, self).__init__()
        self.core = core
        self.lin = nn.Linear(200, 4)

    def forward(self, x):
        x = self.core(x)
        x = self.lin(x)
        return x.view(x.size(0), -1)

def select_action(model, state):
    # start with e-greedy
    epsilon = 0.1
    if np.random.uniform() > epsilon:
        return model(state).data.max(1)[1].numpy()[0]
    else:
        return np.random.randint(4)

np.random.seed(10)
torch.manual_seed(7)

num_models = 10
envs = [gym.make('small-maze-{}-v0'.format(i)) for i in range(num_models)]
mems = [ReplayMemory() for i in range(num_models)]
core = CoreNet()
mods = [HeadNet(core) for i in range(num_models)]
opts = [Adam(mod.parameters()) for mod in mods]
last_state = [env.reset() for env in envs]
total_rewards = [0 for i in range(num_models)]
episode_steps = [0 for i in range(num_models)]
episode_count = [0 for i in range(num_models)]
episode_history = {k:[] for k in range(num_models)}

total_training_steps = 2500
max_steps_per_episode = 300
batch_size = 10
gamma = 0.99

for _ in range(total_training_steps):
    for i in range(num_models):
        
        # set things for particular model
        state = last_state[i]
        model = mods[i]
        memory = mems[i]
        optimizer = opts[i]
        env = envs[i]

        # act 
        state_variable = Variable(torch.from_numpy(state[np.newaxis,:,:,:].astype('float32')))
        action = select_action(model, state_variable)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state)

        # train
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(batch_size)
        action_batch = action_batch.reshape(batch_size,1)
        state_batch = Variable(torch.from_numpy(state_batch.astype('float32')))
        action_batch = Variable(torch.from_numpy(action_batch.astype('int64')))
        reward_batch = Variable(torch.from_numpy(reward_batch.astype('float32')))
        next_state_batch = Variable(torch.from_numpy(next_state_batch.astype('float32')))
        next_state_values = model(next_state_batch).max(1)[0]
        state_action_values = model(state_batch).gather(1, action_batch)
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # handle done
        # set things for model
        if done or episode_steps[i] >= max_steps_per_episode:
            total_rewards[i] += reward * gamma ** episode_steps[i]
            episode_history[i].append(episode_steps[i])
            print('Agent {} finished episode {} in {} steps and {} reward.'
                .format(i, episode_count[i], episode_steps[i], total_rewards[i]))
            episode_steps[i] = 0
            total_rewards[i] = 0
            episode_count[i] += 1
            last_state[i] = env.reset()

        else:
            last_state[i] = np.copy(next_state)
            total_rewards[i] += reward * gamma ** episode_steps[i]
            episode_steps[i] += 1

print(episode_history)
