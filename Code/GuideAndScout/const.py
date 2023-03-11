import torch


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


def verbPrint(string, verbose):
    if getVerbose() >= verbose:
        print(string)


trainSeed = 10
evalSeed = 17
