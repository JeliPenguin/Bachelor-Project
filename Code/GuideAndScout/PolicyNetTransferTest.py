from Runner.RunnerBase import Runner
from joblib import load, dump


saveNames = ["Two4X4", "Two5X5", "Noised5X5"]


def test(saveName):
    fileName = f"./Saves/{saveName}/"
    agents = load(fileName + "agents")
    agentSettings = []
    for a in agents:
        # a._memory.clear()
        agentSettings.append(
            (a._policy_net, a._epsStart, a._epsEnd, a._epsDecay, a._eps_done))

    dump(agentSettings, fileName+"agentSettings")
    # myRun = Runner(envSetting, saveName=saveName)
    # myRun.test(verbose=1)


def testTrained(saveName):
    fileName = f"./Saves/{saveName}/"
    envSetting = load(fileName+"envSetting")
    myRun = Runner(envSetting, saveName=saveName)
    myRun.test(verbose=1)


def editEnv(saveName):
    fileName = f"./Saves/{saveName}/"
    envSetting = load(fileName+"envSetting")
    envSetting["treatNum"] = 2
    # # envSetting["scoutsNum"] = 2
    envSetting["noiseP"] = 0.05
    # envSetting["RAND_EPS"] = 5
    envSetting["TRAIN_EPS"] = 100000

    dump(envSetting, fileName+"envSetting")

    print(envSetting)


# for name in saveNames:
#     test(name)
# testTrained("Two5X5")
editEnv("Two4X4")
