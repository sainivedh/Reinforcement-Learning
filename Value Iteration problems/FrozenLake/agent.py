import random
import numpy as np
random.seed(0)
np.random.seed(0)

class RandomAgent:

    def __init__(self,action_space):
        self.action_space = action_space

    def chooseAction(self):
        return self.action_space.sample()

class ValueIterAgent:

    def __init__(self,env,gamma):
        self.max_iterations = 1000
        self.gamma = gamma
        self.action_space = env.action_space
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.state_prob = env.env.P

        self.values = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states)

    def ValueIter(self):

        for _ in range(self.max_iterations):
            prev_values = np.copy(self.values)
            for state in range(self.num_states):
                Q_value = []
                for action in range(self.num_actions):
                    next_state_action_rewards = []
                    for tran_probs, next_state, reward_probs, _ in self.state_prob[state][action]:
                        next_state_action_rewards.append(tran_probs*(reward_probs+(self.gamma*prev_values[next_state])))
                    Q_value.append(sum(next_state_action_rewards))
                self.values[state] = max(Q_value)
    
    def extract_policy(self):

        for state in range(self.num_states):
            state_actions = np.zeros(self.num_actions)
            for action in range(self.num_actions):
                for tran_probs, next_step, reward_probs, _  in self.state_prob[state][action]:
                    state_actions[action] += (tran_probs*(reward_probs+self.gamma*self.values[next_step]))
            self.policy[state] = np.argmax(state_actions)
    
    def chooseAction(self,state):
        return self.policy[state]

