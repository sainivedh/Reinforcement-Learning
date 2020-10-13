import random
import numpy as np
random.seed(0)
np.random.seed(0)


class Environment:

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self._probs = [random.random() for _ in range(self.num_arms)]
        self.action_space = range(self.num_arms)

    def try_arm(self,arm_num):

        val = random.random() < self._probs[int(arm_num)]

        return 1.0 if val else 0.0

