import pickle
import numpy as np


class Policy_Iteration():
    def __init__(self, env, discount, accuracy):
        self.env = env
        self.state_space = self.env.observation_space.n
        # Initial Deterministic Policy
        self.policy = np.zeros(self.state_space)
        self.action_space = self.env.action_space.n
        self.value = np.zeros(self.state_space)
        self.discount = discount
        self.accuracy = accuracy

    def train(self):
        self.policy_eval()
        self.policy_improve()

    def policy_eval(self):
        while True:
            delta = 0
            for state in range(self.state_space):
                original_value = self.value.item(state)
                # Asynchronously update the value function
                self.value[state] = self.bellman(state, self.policy[state])
                delta = max(abs(original_value-self.value[state]), delta)
            if delta <= self.accuracy:
                # Reached desired accuracy
                break

    def policy_improve(self):
        stable_policy = True
        for state in range(self.state_space):
            old_action = self.policy[state]
            # Take greedy action according to the evaluated value function
            self.policy[state] = self.greedy(state)
            if old_action != self.policy[state]:
                stable_policy = False
        # If all actions are the same as the previous policy, the policy is optimal according to policy improvement theorem
        if stable_policy:
            # print(self.policy)
            pickle.dump(self.policy, open("save.p", "wb"))
            return
        # Otherwise continue the evaluation-improvement cycle
        self.train()

    def greedy(self, state):
        actions = []
        for action in range(self.action_space):
            action_val = self.bellman(state, action)
            actions.append(action_val)
        return np.argmax(actions)

    def bellman(self, state, action):
        # Bellman expectation equation
        dynamics = self.env.P[state][action]
        new_value = 0
        # \pi(a|s) ignored as the policy is deterministic
        for next_step in dynamics:
            probability, nextstate, reward, done = next_step
            new_value += probability * \
                (reward + self.discount * self.value[nextstate])
        return new_value
