from pettingzoo.mpe import simple_spread_v2
from time import sleep

env = simple_spread_v2.env(max_cycles=25,continuous_actions=False,render_mode="human")
env.reset()
#print(env.action_space(env.agent_selection).sample())

def policy(obs,agent):
    aS = env.action_space(agent).sample()
    return aS

for agent in env.agent_iter():
    #env.render()
    observation, reward, done, truncation, info = env.last()
    print(env.truncations)
    action = policy(observation, agent)
    env.step(action)