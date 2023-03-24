from joblib import dump, load
import matplotlib.pyplot as plt
from Runner.Runner import Runner
from Evaluator import Evaluator


envSetting = {
    "noised": False,
}

trainSetting = {
    "method": "Norm",
    "TRAIN_EPS": 10000,
    "batchSize": 128,
    "gamma": 0.99,
    "epsStart": 0.9,
    "epsEnd": 0.05,
    "epsDecay": 12000,
    "tau": 0.005,
    "lr": 1e-4,
    "seed": 10
}


def quickTest():
    envSetting["TRAIN_EPS"] = 2
    envSetting["row"] = 4
    envSetting["column"] = 4
    myRun = Runner("Spread", saveName="Test")
    myRun.train(envSetting, wandbLog=False)
    myRun.test(verbose=1, maxEps=10)


def spreadTrain():
    dim = 4
    envSetting["row"] = dim
    envSetting["column"] = dim
    myRun = Runner("Spread", saveName=f"Spread{dim}X{dim}")
    myRun.train(envSetting, trainSetting)
    # myRun.test(verbose=1)


def evaluate():
    norm = "Two5X5"
    noised = "Noised5X5"
    eT = Evaluator(norm, noised)
    eT.evaluate(False, range=0.6)


def testTrained():
    dim = 3
    envSetting["row"] = dim
    envSetting["column"] = dim
    run = Runner("Spread", "Spread3X3")
    # run.train(envSetting, trainSetting)
    run.test(verbose=1)


if __name__ == "__main__":
    # noisedTest(0.3)
    # noisedTest(0.3)
    # noisedRandomTest()
    # spreadTrain()
    # randomRun()
    # evaluate()
    # spreadTrain()
    testTrained()
