import gym
from main import *
import numpy as np

env = gym.make('CartPole-v1')
# print(env.observation_space.high[0])
# print(env.observation_space.low[0])
'''Ignoring cart velocity and pole angular velocity for now to reduce complexity'''
discreteCartPosSize = 50
discreteCartVSize = 25
discretePolePosSize = 200
discretePoleVSize = 200
DISCRETE_OS_SIZE = [discreteCartPosSize,discreteCartVSize,discretePolePosSize,discretePoleVSize]
cart_pos_range = env.observation_space.high[0] - env.observation_space.low[0]
#Assumes range -10 - 10
cart_v_range = 20
cart_v_low = -10
pole_pos_range = env.observation_space.high[2] - env.observation_space.low[2]
#Assumes range -10 - 10
pole_v_range = 20
pole_v_low = -10
discrete_os_win_size = (np.array([cart_pos_range,cart_v_range,pole_pos_range,pole_v_range])) / DISCRETE_OS_SIZE
discount = 0.9
learning_rate = 0.1
epsilon = 1
epsilon_decay = 0.99999
#sample_run(env,20)

q_table = np.random.uniform(low = -2, high= 0,size = (discreteCartPosSize,discreteCartVSize,discretePolePosSize,discretePoleVSize,env.action_space.n))
num_episode = 50000

def normalise_state(s):
    """print(f"Cart pos: {s[0]} Pole pos: {s[2]} Pole v:{s[3]}")"""
    state = (s[0] - env.observation_space.low[0],s[1]-cart_v_low,s[2]-env.observation_space.low[2],s[3] - pole_v_low)
    discrete_state = state / discrete_os_win_size
    #print(discrete_state)
    # discrete_state[3] = max(0,min(discretePoleVSize-1,discrete_state[3]))
    # discrete_state[1] = max(0,min(discreteCartVSize-1,discrete_state[1]))
    #print(state,discrete_state)
    return tuple(discrete_state.astype(np.int64))

def choose_action(s,q_table):
    return np.argmax(q_table[s])

def choose_action_eps(s,q_table):
    p = np.random.random()
    if p < epsilon:
      return env.action_space.sample()
    actionVal = [q_table[s][action] for action in range(env.action_space.n)]
    return np.argmax(actionVal)

def maxQ(s_prime):
    return max(q_table[s_prime])

def train(epsilon):
    for eps in range(num_episode):
        state = normalise_state(env.reset()[0])
        done = False
        while not done:
            action = choose_action_eps(state,q_table)
            next_state, reward, done, _,_ = env.step(action)
            if not done:
                next_state = normalise_state(next_state)
                q_error = reward + discount * maxQ(next_state) - q_table[state][action]
                q_table[state][action] = q_table[state][action] + learning_rate * q_error 
            state = next_state
        if epsilon > 0.05 and (eps % 100 == 0):
            epsilon = epsilon * epsilon_decay

def run():
    env = gym.make('CartPole-v1',render_mode="human")
    qt = pickle.load(open("q_table.p", "rb"))
    print(qt.shape)
    state = normalise_state(env.reset()[0])
    print(state)
    done = False
    while not done:
        env.render()
        action = choose_action(state,qt)
        state, _, done, _,_ = env.step(action)
        state = normalise_state(state) 


# train(epsilon)
# pickle.dump(q_table, open("q_table.p", "wb"))
run()