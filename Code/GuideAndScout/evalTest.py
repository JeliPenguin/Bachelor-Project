from Runner.EvalRunner import EvalRunner


myRun = EvalRunner(saveName="Two5x5")
print(myRun._currentSeed)
myRun.test(noiseLevel=1)
