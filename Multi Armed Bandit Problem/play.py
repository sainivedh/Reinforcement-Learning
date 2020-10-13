import random
import numpy as np
from environments_sai import Environment
from agents_sai import RandomAgent
from agents_sai import ValueApproxAgent


num_gen = 1000
tot_reward = 0

env = Environment(6)
agent = ValueApproxAgent(env.action_space,0.05)

for i in range(num_gen):
    curr_arm = agent.choose_action()
    curr_reward = env.try_arm(curr_arm)
    agent.learn(curr_arm,curr_reward)
    tot_reward += curr_reward

print('Total Reward: ',tot_reward)
print('Original Probabilities: ',env._probs)
print('Computed Probabilities: ',agent.approx_values)