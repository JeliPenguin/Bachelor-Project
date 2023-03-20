from joblib import dump, load
import matplotlib.pyplot as plt
from Runner.NormRunner import Runner
from Runner.SchedRunner import SchedRunner
from Evaluator import Evaluator


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
    "noised": False,
    "scoutsNum": 2,
    "TEST_MAX_EPS": 20,
}


def quickTest():
    envSetting["TRAIN_EPS"] = 2
    envSetting["TEST_MAX_EPS"] = 5
    myRun = Runner("Spread", saveName="Test")
    myRun.train(envSetting,wandbLog=False)
    myRun.test(verbose=2)


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

def evaluate():
    norm = "Two5X5"
    noised = "Noised5X5"
    eT = Evaluator(norm, noised)
    eT.evaluate(False,range=0.6)


if __name__ == "__main__":
    # noisedTest(0.3)
    quickTest()
    # randomRun()
    # actualRun()
    # evaluate()
    # schedTrain()
