from pettingzoo.mpe import simple_speaker_listener_v3
from time import sleep
env = simple_speaker_listener_v3.env(max_cycles=25,continuous_actions=False,render_mode="human")
env.reset()
#print(env.action_space(env.agent_selection).sample())

def policy(obs,agent):
    aS = env.action_space(agent).sample()
    return aS

for agent in env.agent_iter():
    env.render()
    observation, reward, done, truncation, info = env.last()
    action = policy(observation, agent)
    env.step(action)
    sleep(1)