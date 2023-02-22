import torch

EMPTY = "-"
TREAT = "$"

UP = 0
LEFT = 1
RIGHT = 2
DOWN = 3
STAY = 4
ACTIONSPACE = ["UP", "LEFT", "RIGHT", "DOWN", "STAY"]
def decodeAction(num: int):
    mapping = {
        0: (-1, 0),
        1: (0, -1),
        2: (0, 1),
        3: (1, 0),
        4: (0, 0)
    }
    return mapping[num]

def tupleAdd(xs, ys): return tuple(x + y for x, y in zip(xs, ys))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# 0 to hide all information
# 1 for basic render (the environment, agent infomration)
# 2 for agent communication information
# 3 for checksum information and message history
VERBOSE = 0
def setVerbose(num):
    global VERBOSE
    VERBOSE = num


def getVerbose():
    return VERBOSE
