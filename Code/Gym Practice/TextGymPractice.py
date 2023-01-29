import gym
from main import run

#Practicing Policy/Value iteration using Textual Gyms

env = gym.make("FrozenLake-v1", render_mode="human", desc=None, map_name="4x4",
               is_slippery=False)
run(env)