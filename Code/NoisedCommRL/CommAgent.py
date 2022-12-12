from const import *

import numpy as np



class CommAgent(GridAgent):
    def __init__(self,id, obs_dim,actionSpace, learning_rate=0.1, discount=0.9, epsilon_decay=0.9) -> None:
        super().__init__(id, obs_dim,actionSpace, learning_rate, discount, epsilon_decay)

class ScoutAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace, learning_rate=0.1, discount=0.9, epsilon_decay=0.9) -> None:
        super().__init__(id, obs_dim, actionSpace, learning_rate, discount, epsilon_decay)
    
    def choose_action(self, s, eps=False):
        return self.choose_random_action()
    

class GuideAgent(CommAgent):
    def __init__(self, id, obs_dim, actionSpace, learning_rate=0.1, discount=0.9, epsilon_decay=0.9) -> None:
        super().__init__(id, obs_dim, actionSpace, learning_rate, discount, epsilon_decay)
    
    def choose_action(self, s):
        return STAY
    
