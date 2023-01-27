import torch
import numpy as np
from typing import *


def addNoise(msg, p=0.01):
    noise = np.random.random(msg.shape) < p
    noiseAdded = []
    for m, n in zip(msg, noise):
        if n == 0:
            noiseAdded.append(m)
        else:
            noiseAdded.append(1-m)
    noiseAdded = np.array(noiseAdded)
    print(noiseAdded)
    return (noiseAdded)


def encoder(msg: List[int]):
    formatted = np.array(msg, dtype=np.uint8)
    encoded = np.unpackbits(formatted)

    return encoded


def decoder(msg):
    decoded = np.packbits(msg)
    return decoded


def simulateChannel():
    msg = [1, 2, 3]
    encoded = encoder(msg)
    encoded = addNoise(encoded)
    decoded = decoder(encoded)
    print(decoded)


simulateChannel()
