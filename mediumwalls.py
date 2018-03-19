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
    def __init__(self, size=400, shape=(5, 5, 8)):
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
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(4)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        return x

class MediumNet(nn.Module):
    def __init__(self, core):
        super(MediumNet, self).__init__()
        self.core = core
        self.conv1 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(1)
        self.lin1 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.core(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.lin1(x.view(x.size(0), 100))
        return x.view(x.size(0), -1)

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(1)
        self.lin = nn.Linear(100, 4)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.lin(x.view(x.size(0), 100))
        return x.view(x.size(0), -1)

class LinearBaselineNet(nn.Module):
    def __init__(self):
        super(LinearBaselineNet, self).__init__()
        self.lin = nn.Linear(800, 4)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
        x = self.lin(x.view(x.size(0), 800))
        return x.view(x.size(0), 4)

def select_action(model, state):
    # start with e-greedy
    epsilon = 0.1
    if np.random.uniform() > epsilon:
        return model(state).data.max(1)[1].numpy()[0]
    else:
        return np.random.randint(4)

np.random.seed(10)
torch.manual_seed(7)

env = gym.make('medium-maze-3-v0')

for i_maze in range(10):
    env = gym.make('medium-maze-{}-v0'.format(i_maze))

    memory = ReplayMemory(size=2000, shape=(10,10,8))
    
    # proposed method
    core = CoreNet()
    core.load_state_dict(torch.load('teststate'))
    model = MediumNet(core)
    params = list(model.conv1.parameters()) + list(model.lin1.parameters()) + list(model.relu1.parameters()) + list(model.bn1.parameters())
    optimizer = Adam(params)
    
    
    num_episodes = 10
    max_steps_per_episode = 200
    batch_size = 10
    gamma = 0.99
    
    for i_episode in range(num_episodes):
        steps = 0
        done = False
        state = env.reset()
        while not done and steps < max_steps_per_episode:
            # act
            model.eval()
            state_variable = Variable(torch.from_numpy(state[np.newaxis,:,:,:].astype('float32')))
            action = select_action(model, state_variable)
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state)
    
            # env.render()
            
            # train
            model.train()
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
            steps += 1
        print(env._unwrapped.maze_view.walk_into_wall / float(steps))
    
    model = BaselineNet()
    optimizer = Adam(model.parameters())
    
    num_episodes = 10
    max_steps_per_episode = 500
    batch_size = 10
    gamma = 0.99
    print('Baseline')
    memory = ReplayMemory(size=2000, shape=(10,10,8))
    
    
    for i_episode in range(num_episodes):
        steps = 0
        done = False
        state = env.reset()
        while not done and steps < max_steps_per_episode:
            # act
            model.eval()
            state_variable = Variable(torch.from_numpy(state[np.newaxis,:,:,:].astype('float32')))
            action = select_action(model, state_variable)
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action, reward, next_state)
    
            # env.render()
            
            # train
            model.train()
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
            steps += 1
        print(env._unwrapped.maze_view.walk_into_wall / float(steps))
    
    # if i_episode % 10 == 0:
        # torch.save(core.state_dict(), 'medstate_proposed')
        # torch.save(core.state_dict(), 'medstate_baseline')

