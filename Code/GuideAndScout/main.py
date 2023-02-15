from Runner import Runner
from const import setVerbose
"""
Gridworld with treats, a Guide agent and a Scout agent

Guide agent cannot move but can observe the environment and send messages

Scout agent can move but cannot observe the environment and send messages

Guide and scout need to cooporate through communication to obtain all treats

Communication currently with no noise added
"""


"""
defaultEnvSetting = {
    "row": 5,
    "column": 5,
    "treatNum": 2,
    "scoutsNum": 1,
    "noised": False,
    "TRAIN_EPS": 5,
    "TEST_MAX_EPS": 30,
    "RAND_EPS": 1,
}
"""


if __name__ == "__main__":
    envSetting = {
        "row": 5,
        "column": 5,
        "scoutsNum": 2,
        "RAND_EPS": 1,
        "TEST_MAX_EPS": 30,
        "TRAIN_EPS": 5000,
    }

    myRun = Runner(envSetting, saveName="DoubleScout")
    # myRun.randomRun()
    setVerbose(0)
    # myRun.train()
    setVerbose(2)
    myRun.test()
