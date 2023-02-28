from Runner import Runner
from joblib import load, dump


def test():
    saveName = "Noised5x5"
    fileName = f"./Saves/{saveName}/"
    envSetting = load(fileName+"envSetting")
    print(envSetting)
    agents = load(fileName + "agents")
    print(agents)
    policyNets = []
    for a in agents:
        print(a._policy_net)
        policyNets.append(a._policy_net)

    dump(policyNets, fileName+"policyNets")
    # myRun = Runner(envSetting, saveName=saveName)
    # myRun.test(verbose=1)


def testTrained():
    saveName = "Noised5x5"
    fileName = f"./Saves/{saveName}/"
    envSetting = load(fileName+"envSetting")
    policyNets = load(fileName + "policyNets")
    myRun = Runner(envSetting, saveName=saveName)
    myRun.test(verbose=1)

# testTrained()
