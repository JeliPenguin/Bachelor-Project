from Runner.NormRunner import Runner


def testRun(run, noiseLevel, noiseHandlingMode):

    steps, reward = run.test(
        verbose=1, noiseLevel=noiseLevel, noiseHandlingMode=noiseHandlingMode)
    return steps, reward


runner = Runner("Noised5x5", eval=True)
for noise in range(2):
    for _ in range(2):
        testRun(runner, 0, None)
