from joblib import dump, load
import matplotlib.pyplot as plt
from Runner.NormRunner import Runner
from Runner.SchedRunner import SchedRunner
from Evaluator import Evaluator
"""
Gridworld with treats, a Guide agent and Scout agent(s)

Guide agent cannot move but can observe the environment and send messages

Scout agent can move but cannot observe the environment and send messages

Guide and scout need to cooporate through communication to obtain all treats

With additional scouts added, scouts themselves would also need to cooperate to obtain all treats in least amount
of time
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
    "row": 5,
    "column": 5,
    "RAND_EPS": 5,
    "noised": True,
    "scoutsNum": 2,
    "TEST_MAX_EPS": 20,
}


def quickTest():
    envSetting["TRAIN_EPS"] = 2
    envSetting["TEST_MAX_EPS"] = 10
    myRun = Runner(envSetting, saveName="Test")
    myRun.train(wandbLog=False)
    myRun.test(verbose=3)


def noisedRandomTest():
    envSetting["noised"] = True
    envSetting["noiseP"] = 0.05
    myRun = Runner(saveName="Test2")
    myRun.randomRun(verbose=3)


def noisedTest(p):
    envSetting["TRAIN_EPS"] = 1
    envSetting["TEST_MAX_EPS"] = 20
    envSetting["noised"] = True
    envSetting["noiseP"] = p
    myRun = Runner(saveName="Test2")
    myRun.train(envSetting=envSetting, wandbLog=False)
    myRun.test(noiseHandlingMode=0, verbose=-1)


def randomRun():
    myRun = Runner(envSetting, saveName="Test")
    myRun.randomRun(verbose=2)


# def actualRun():
#     envSetting["TRAIN_EPS"] = 100000
#     myRun = Runner(envSetting, saveName="Noised5X5")
#     myRun.train(wandbLog=False)
#     myRun.test(verbose=1)


def schedTrain():
    envSetting["TRAIN_EPS"] = 100000
    # envSetting["TRAIN_EPS"] = 10
    myRun = SchedRunner(saveName="Sched5X5")
    myRun.train(envSetting=envSetting)
    # myRun.test(verbose=1)


def testTrained():
    myRun = Runner(saveName="Two5X5")
    myRun.test(verbose=3)


def plotter():
    norm = load("./Saves/Two5X5/episodicRewards")
    noised = load("./Saves/Noised5X5/episodicRewards")
    plt.plot(norm)
    plt.plot(noised)
    plt.show()


def evaluate():
    norm = "Two5X5"
    noised = "Noised5X5"
    eT = Evaluator(norm, noised)
    eT.evaluate(True)


if __name__ == "__main__":
    # noisedTest(0.3)
    # noisedRandomTest()
    # quickTest()
    # randomRun()
    # actualRun()
    # testTrained()
    # plotter()
    evaluate()
    # schedTrain()
