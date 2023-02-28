from Agents.DQN import DQNAgent
from Environment.CommChannel import CommChannel
import numpy as np
import torch
from const import *
from ErrorDetection.Checksum import Checksum


class CommAgent(DQNAgent):
    def __init__(self, id, obs_dim, actionSpace, noiseHandling=False, batchSize=128, gamma=1, epsStart=0.9, epsEnd=0.05, epsDecay=1000, tau=0.005, lr=0.0001) -> None:
        self._agentNum = obs_dim[0]
        self._totalTreatNum = obs_dim[1]
        n_observations = 2 * (self._agentNum + self._totalTreatNum)
        super().__init__(id, n_observations, actionSpace,
                         batchSize, gamma, epsStart, epsEnd, epsDecay, tau, lr)

        self.errorDetector = Checksum(8)
        self._majorityNum = 3
        self._noiseHandling = noiseHandling
        self.reset()

    def reset(self):
        self._messageReceived = {}

        self._messageSent = {}
        self._action = None
        self._messageMemory = {
            "state": None,
            "reward": None,
            "sPrime": None
        }
        self._majorityMem = []

    def setNoiseHandling(self, noiseHandlingMode):
        self._noiseHandlingMode = noiseHandlingMode
        self._noiseHandling = noiseHandlingMode is not None

    def setChannel(self, channel: CommChannel):
        self._channel = channel
        self.reset()

    def tensorize(self, msg):
        stateTensor = None
        actionTensor = None
        sPrimeTensor = None
        rewardTensor = None
        for tag, content in msg.items():
            if content is not None:
                if tag == "action":
                    actionTensor = torch.tensor(
                        [content], dtype=torch.int64, device=device)
                elif tag == "state":
                    stateTensor = torch.tensor(content, dtype=torch.float32,
                                               device=device).unsqueeze(0)
                elif tag == "sPrime":
                    sPrimeTensor = torch.tensor(content, dtype=torch.float32,
                                                device=device).unsqueeze(0)
                elif tag == "reward":
                    rewardTensor = torch.tensor(
                        content, dtype=torch.float32, device=device)
        return stateTensor, actionTensor, sPrimeTensor, rewardTensor

    def encodeMessage(self):
        """
        Message Order: State - Reward - sPrime each as unsigned 8 bits
        For rewards, unsigned 129-255 used to represents -127 - -1
        """
        if self._messageMemory["reward"] is None and self._messageMemory["sPrime"] is None:
            # Case state only
            msgString = self._messageMemory["state"]
        elif self._messageMemory["sPrime"] is None:
            # Case termination
            msgString = np.concatenate(
                (self._messageMemory["state"], self._messageMemory["reward"]))
        else:
            msgString = np.concatenate(
                (self._messageMemory["state"], self._messageMemory["reward"], self._messageMemory["sPrime"]))
        formatted = np.array(msgString, dtype=np.uint8)
        encoded = np.unpackbits(formatted)
        return encoded

    def clearPreparedMessage(self):
        self._messageMemory = {
            "state": None,
            "reward": None,
            "sPrime": None
        }

    def prepareMessage(self, msg, tag: str):
        # Staging the message to be sent
        self._messageMemory[tag] = msg

    def rememberAction(self, action):
        # Agent itself remebers the action being executed
        self._action = action

    def stringify(self, encoded):
        encodedString = ""
        for b in encoded:
            encodedString += str(b)
        return encodedString

    def sendMessage(self, recieverID: int):
        pass

    def decodeMessage(self, encodedMsg):
        msgLen = len(encodedMsg)
        obsLen = self._n_observations
        parse = {
            "state": None,
            "reward": None,
            "sPrime": None
        }
        parse["state"] = encodedMsg[:obsLen]
        if msgLen > self._n_observations:
            parse["reward"] = [encodedMsg[obsLen]]
            if msgLen > self._n_observations + 1:
                parse["sPrime"] = encodedMsg[obsLen+1:]
        if parse["reward"] is not None and parse["reward"][0] > 129:
            parse["reward"] = [parse["reward"][0]-256]
        return parse

    def storeRecievedMessage(self, senderID, parse, correctChecksum=True):
        # Action independent of the message as agent itself knows what action has been executed
        # Policy assumed to be a deterministic policy
        parse["action"] = self._action
        verbPrint(f"Message Received: {parse}\n", 2)
        for tag, content in parse.items():
            if senderID not in self._messageReceived:
                self._messageReceived[senderID] = {tag: content}
            else:
                self._messageReceived[senderID][tag] = (content)

    def recieveNoisyMessage(self, senderID: int, msg):
        pass

    def recieveNormalMessage(self, senderID: int, msg):
        msg = np.packbits(msg)
        decoded = self.decodeMessage(msg)
        self.storeRecievedMessage(senderID, decoded)

    def recieveMessage(self, senderID: int, msg):
        if self._noiseHandling:
            self.recieveNoisyMessage(senderID, msg)
        else:
            self.recieveNormalMessage(senderID, msg)

    def broadcastSignal(self, signal):
        self._channel.broadcastSignal(self._id, signal)

    def recieveBroadcast(self, signal):
        pass
