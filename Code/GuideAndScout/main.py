from Runner import Runner
import sys
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
        "TEST_MAX_EPS": 1,
        "TRAIN_EPS": 2,
        "noised": True
    }

    if len(sys.argv) > 1:
        envSetting["TRAIN_EPS"] = int(sys.argv[1])

    myRun = Runner(envSetting, saveName="DoubleScout")
    # myRun.randomRun()
    # myRun.train(log=False)
    myRun.test()
