from Runner.EvalRunner import EvalRunner


def reEvaluate(run, noiseLevel, noiseHandlingMode):
    steps, reward = run.test(
        verbose=1, noiseLevel=noiseLevel, noiseHandlingMode=noiseHandlingMode)
    return steps, reward

def evaluate():
    run= EvalRunner("Test")
    for noise in range(2):
        for _ in range(2):
            reEvaluate(run, 0, None)

evaluate()
