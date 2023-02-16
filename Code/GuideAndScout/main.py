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
            "scoutsNum": 2,
            "noised": False,
            "noiseP": 0.01,
            "TRAIN_EPS": 5,
            "TEST_MAX_EPS": 30,
            "RAND_EPS": 1,
}
"""


if __name__ == "__main__":
    envSetting = {
        "row": 3,
        "column": 3,
        "RAND_EPS": 1,
        "noised": False,
        "TEST_MAX_EPS": 5,
        "TRAIN_EPS": 2,
    }

    myRun = Runner(envSetting, saveName="DoubleScout")
    # myRun.randomRun()
    setVerbose(0)
    myRun.train(log=False)
    setVerbose(3)
    myRun.test()
