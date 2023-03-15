import numpy as np


class CommChannel():
    def __init__(self, agents, noiseP: float, noised=False) -> None:
        self._noised = noised
        self.setNoiseP(noiseP)
        self._agents = agents

    def setupChannel(self):
        for agent in self._agents:
            agent.setChannel(self)

    def setNoiseP(self, noiseP:float):
        # print(f"Setting noisep: {noiseP}")
        self._noiseP = noiseP

    def addNoise(self, msg):
        noise = np.random.random(msg.shape) < self._noiseP
        noiseAdded = []
        for m, n in zip(msg, noise):
            if n == 0:
                noiseAdded.append(m)
            else:
                noiseAdded.append(1-m)
        noiseAdded = np.array(noiseAdded)
        return (noiseAdded)

    def broadcastSignal(self, senderID, signal):
        if self._noised:
            signal = self.addNoise(signal)
        for id in range(len(self._agents)):
            if id != senderID:
                self._agents[id].recieveBroadcast(signal)

    def sendMessage(self, senderID, receiverID, msg):
        receiver = self._agents[receiverID]
        if self._noised:
            msg = self.addNoise(msg)

        receiver.recieveMessage(senderID, msg)
