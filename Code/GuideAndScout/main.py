import os
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
}


def quickTest():
    trainSetting["TRAIN_EPS"] = 2
    envSetting["row"] = 3
    envSetting["column"] = 3
    myRun = Runner("Spread", saveName="Test")
    myRun.train(envSetting, trainSetting)
    myRun.test(verbose=3, maxEps=10)


def spreadTrain():
    dim = 4
    envSetting["row"] = dim
    envSetting["column"] = dim
    trainSetting["TRAIN_EPS"] = 150000
    trainSetting["epsDecay"] = 14000
    myRun = Runner("Spread", saveName=f"Spread{dim}X{dim}")
    myRun.train(envSetting, trainSetting, showInterTrain=True)


def evaluate():
    eT = Evaluator(envType="FindingTreat")
    # eT = Evaluator(envType="Spread")
    # eT.evaluate()
    eT.plotAll()
    # eT.plotBest()
    # eT.normNoiseCompare()


def hyperParamTune():
    lr = [1e-4, 1e-3]
    batchSize = [32, 64, 128, 256]
    epsDecay = [8000, 10000, 12000, 14000, 16000, 20000]
    tau = [0.0005, 0.001, 0.005, 0.01, 0.05]
    # env = "FindingTreat"
    env = "Spread"
    if env == "Spread":
        dim = 3
        envSetting["row"] = dim
        envSetting["column"] = dim
        trainSetting["TRAIN_EPS"] = 150000
        batchSize = [64, 128, 256]
        epsDecay = [14000, 16000, 18000, 20000]
    for l in lr:
        for b in batchSize:
            for e in epsDecay:
                for t in tau:
                    runName = f"HyperParam/{env}/{env}_{l}_{b}_{e}_{t}"
                    if not os.path.exists("./Saves/"+runName):
                        trainSetting["batchSize"] = b
                        trainSetting["lr"] = l
                        trainSetting["epsDecay"] = e
                        trainSetting["tau"] = t
                        run = Runner(env, runName)
                        run.train(envSetting, trainSetting)


if __name__ == "__main__":
    # spreadTrain()
    evaluate()
    # spreadTrain()
    # testTrained()
    # hyperParamTune()
    # quickTest()
