import pickle
from policy_iteration import Policy_Iteration
from value_iteartion import Value_Iteration


def do_train(env):
    discount_factor = 0.9
    acc = 0.0001
    agent = Policy_Iteration(env, discount_factor, acc)
    #agent = Value_Iteration(env,discount_factor,acc)
    agent.train()


def run(env):
    do_train(env)
    policy = pickle.load(open("save.p", "rb"))
    env.reset()
    state = 0
    done = False
    while not done:
        env.render()
        action = int(policy[state])
        state, reward, done, _, _ = env.step(action)
    env.close()


def sample_run(env, eps_num):
    for i_episode in range(eps_num):
        state = env.reset()
        for t in range(20000):
            env.render()
            action = env.action_space.sample()
            state, reward, done, _, info = env.step(action)
            print(
                f"Action: {action}    State: {state}  Reward: {reward}    Done: {done}")
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


