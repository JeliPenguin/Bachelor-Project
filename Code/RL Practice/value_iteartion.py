import pickle
import numpy as np
import gym

class Value_Iteration():
    def __init__(self,env,discount,accuracy):
        self.env = env
        self.state_space = self.env.observation_space.n
        # Initial Deterministic Policy
        self.policy = np.zeros(self.state_space)
        self.action_space = self.env.action_space.n
        self.value = np.zeros(self.state_space)
        self.discount = discount
        self.accuracy = accuracy

    def train(self):
        self.value_iter()
        self.generate_policy()

    def value_iter(self):
        while True:
            delta = 0
            for state in range(self.state_space):
                original_value = self.value.item(state)
                #Asynchronously update the value function
                for action in range(self.action_space):
                    self.value[state] = max(self.bellman(state,action),self.value[state])
                delta = max(abs(original_value-self.value[state]),delta)
            if delta < self.accuracy:
                #Reached desired accuracy
                break

    def bellman(self,state,action):
        #Bellman optimality equation
        dynamics = self.env.P[state][action]
        new_value = 0
        for next_step in dynamics:
            probability, nextstate, reward, done = next_step
            new_value += probability * (reward + self.discount * self.value[nextstate])
        return new_value

    def generate_policy(self):
        for state in range(self.state_space):
            #Take greedy action according to the evaluated value function
            self.policy[state] = self.greedy(state)
        pickle.dump(self.policy, open("save.p", "wb"))

    def greedy(self, state):
        best_action = -1
        best_action_val = -np.inf
        for action in range(self.action_space):
            action_val = self.bellman(state,action)
            if action_val > best_action_val:
                best_action = action
                best_action_val = action_val
        return best_action
