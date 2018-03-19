import tensorflow as tf
import numpy as np
import os
import gym
import time
import sklearn
import itertools
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


env = gym.envs.make("MountainCarContinuous-v0")
video_dir = os.path.abspath("./videos")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
env = gym.wrappers.Monitor(env, video_dir, force=True)

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
# featurizer = sklearn.pipeline.FeatureUnion([
#     ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#     ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#     ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#     ("rbf4", RBFSampler(gamma=0.5, n_components=100))
# ])
# featurizer.fit(scaler.transform(observation_examples))


import matplotlib.pylab as plt

plt.plot(np.array(observation_examples)[:,0])
plt.show()

plt.plot(np.array(observation_examples)[:,1])
plt.show()