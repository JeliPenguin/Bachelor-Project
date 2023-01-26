import torch


def addNoise(msg, p=0.005):
    pass


def encoder(msg, type):
    """
    Encoding Scheme
    - First 2 bits indicate message type
       = 0: state
       = 1: sPrime
       = 2: action
       = 3: reward
    - Following
    """
    typeMap = {
        "state": 0,
        "sPrime": 1,
        "action": 2,
        "reward": 3
    }
    encoded = format(typeMap[type], "02b")
    encoded += msg

    return encoded


def decoder(msg):
    pass


print(torch.tensor([1, 2, 3]))
