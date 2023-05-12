import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

VERBOSE = 0


def setVerbose(num):
    global VERBOSE
    VERBOSE = num


def getVerbose():
    return VERBOSE


def verbPrint(string, verbose):
    if getVerbose() >= verbose:
        print(string)


evalNoiseLevels = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
