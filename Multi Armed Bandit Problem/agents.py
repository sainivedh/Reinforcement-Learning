import random
import numpy as np
np.random.seed(0)
random.seed(0)

class RandomAgent:

    def __init__(self,action_space):
        self.action_space = action_space

    def choose_action(self):
        return np.random.choice(self.action_space)

class ValueApproxAgent:

    def __init__(self,action_space,epsilon=0.05):
        self.action_space = action_space
        self.epsilon = epsilon
        self.approx_values = [0.0] * len(action_space)
        self.observation_counts = [0] * len(action_space)

    def choose_action(self):
        if random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.approx_values)

    def learn(self,action,reward):
        self.observation_counts[action] += 1
        step_size = 1.0/self.observation_counts[action]
        curr_val = self.approx_values[action]
        self.approx_values[action] = curr_val + step_size * (reward - curr_val)

