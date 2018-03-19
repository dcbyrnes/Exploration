
import gym
import gym_maze

import matplotlib.pyplot as plt
import time
i = 1
env = gym.make('very-big-maze-{}-v0'.format(i))
env.render()
env._unwrapped.save_img('/Users/user/Desktop/vbmaze{}.png'.format(i))

# for i in range(5):
# 	env = gym.make('small-maze-{}-v0'.format(i))
# 	env.render()
# 	env._unwrapped.save_img('/Users/user/Desktop/smallmaze{}.png'.format(i))
