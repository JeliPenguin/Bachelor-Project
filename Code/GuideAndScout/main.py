from joblib import dump, load
import matplotlib.pyplot as plt
from Runner.Runner import Runner
from Evaluator import Evaluator


envSetting = {
    "noised": False,
}

trainSetting = {
    "method": "Norm",
    "TRAIN_EPS": 100000,
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
    eT = Evaluator()
    eT.evaluate(True)


def testTrained():
    run = Runner("FindingTreat", "Two5X5")
    # run.train(envSetting, trainSetting)
    run.test(verbose=1)


if __name__ == "__main__":
    # spreadTrain()
    # randomRun()
    evaluate()
    # spreadTrain()
    # testTrained()
