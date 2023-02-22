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

envSetting = {
    "row": 4,
    "column": 4,
    "RAND_EPS": 5,
    "noised": False,
    "noiseP": 0.05,
    "scoutsNum": 2,
    "TEST_MAX_EPS": 20,
}


def quickTest():
    envSetting["TRAIN_EPS"] = 2
    envSetting["TEST_MAX_EPS"] = 5
    myRun = Runner(envSetting, saveName="Test")
    myRun.train(wandbLog=False)
    myRun.test(verbose=4)


def noisedTest():
    envSetting["TRAIN_EPS"] = 2
    envSetting["TEST_MAX_EPS"] = 5
    envSetting["noised"] = True
    envSetting["noiseP"] = 0.1
    myRun = Runner(envSetting, saveName="Test")
    myRun.train(wandbLog=False)
    myRun.test(verbose=4)


def randomRun():
    myRun = Runner(envSetting, saveName="Test")
    myRun.randomRun(verbose=2)


def actualRun():
    envSetting["TRAIN_EPS"] = 100000
    myRun = Runner(envSetting, saveName="Two")
    # myRun.train(wandbLog=False)
    myRun.test(verbose=1)


if __name__ == "__main__":
    noisedTest()
    # quickTest()
    # randomRun()
    # actualRun()
