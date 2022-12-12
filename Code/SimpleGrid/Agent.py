import numpy as np


class GridAgent():
    def __init__(self, id, obs_dim,actionSpace, learning_rate=0.1, discount=0.9, epsilon_decay=0.9) -> None:
        self.id = id
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.actionSpace = actionSpace
        self.actionSpace_n = len(actionSpace)
        self.q_table = np.random.uniform(
            low=-2, high=0, size=obs_dim+(self.actionSpace_n,))

    def policy(s):
        pass

    def maxQVal(self, s):
        return max(self.q_table[s])

    def choose_greedy_action(self, s) -> int:
        return np.argmax(self.q_table[s])

    def choose_eps_action(self, s) -> int:
        p = np.random.random()
        if p < self.epsilon:
            return np.random.randint(self.actionSpace_n)
        return self.choose_greedy_action(s)

    def choose_random_action(self):
        return np.random.randint(0,self.actionSpace_n)
