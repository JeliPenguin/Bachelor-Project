from run import Run

"""
Gridworld with treats, a Guide agent and a Scout agent

Guide agent cannot move but can observe the environment and send messages

Scout agent can move but cannot observe the environment and send messages

Guide and scout need to cooporate through communication to obtain all treats

Communication currently with no noise added
"""

envInfo = {
    "row": 5,
    "column": 5,
    "treatNum": 2,
    "noised": False
}

myRun = Run(envInfo)

# myRun.randomRun()
# myRun.test()
myRun.train()
