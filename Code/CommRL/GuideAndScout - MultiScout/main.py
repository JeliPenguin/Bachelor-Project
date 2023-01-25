from run import Run

"""
Gridworld with treats, a Guide agent and a Scout agent

Guide agent cannot move but can observe the environment and send messages

Scout agent can move but cannot observe the environment and send messages

Guide and scout need to cooporate through communication to obtain all treats

Communication currently with no noise added
"""

envInfo = {
    "row": 8,
    "column": 8,
    "treatNum": 2,
    "noised": False,
    "TRAIN_EPS": 10,
    "TEST_MAX_EPS": 30,
    "RAND_EPS": 3,
}

myRun = Run(envInfo)

# myRun.randomRun()

myRun.train()
myRun.test()
