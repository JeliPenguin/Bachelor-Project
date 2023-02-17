import torch

EMPTY = "-"
TREAT = "$"

UP = 0
LEFT = 1
RIGHT = 2
DOWN = 3
STAY = 4
ACTIONSPACE = ["UP", "LEFT", "RIGHT", "DOWN", "STAY"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ",device)


VERBOSE = 0


def setVerbose(num):
    global VERBOSE
    VERBOSE = num


def getVerbose():
    return VERBOSE
