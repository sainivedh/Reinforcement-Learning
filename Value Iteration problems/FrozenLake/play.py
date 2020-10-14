import gym
from agent import RandomAgent
from agent import ValueIterAgent
import numpy as np
env = gym.make('FrozenLake-v0')
gamma = 1

#Step1: Instantiate a Random/ValueIter Agent

Randomagent = RandomAgent(env.action_space)
agent = ValueIterAgent(env,gamma)
agent.ValueIter()
agent.extract_policy()
#Step2: For Value Iter Agent, Evaluate Policy

#Step3: Play Frozen Lake 1000 times with this policy and measure rewards
all_rewards_ValueIter = []
all_rewards_Random = []

for episode in range(1000):
    obs = env.reset()
    while True:
        action_ValueIter = agent.chooseAction(obs)
        obs,reward,done,info = env.step(action_ValueIter)
        if done:
            all_rewards_ValueIter.append(reward)
            break

    obs = env.reset()
    while True:
        action_Random = Randomagent.chooseAction()
        obs,reward,done,info = env.step(action_Random)
        if done:
            all_rewards_Random.append(reward)
            break
    
        
        


#Step4: Print Average Reward	
print('Total Reward for Random Agent: ',sum(all_rewards_Random))    	
print('Total Reward for Value Iteration Agent: ', sum(all_rewards_ValueIter))
   	

