class CommChannel():
    def __init__(self,agentNum,noised=False) -> None:
        self.noised = noised
        self.messages = {}

    def addNoise(self,message):
        noisedMsg = None

        return noisedMsg

    def encodeMessage(self,msg):
        pass

    def sendMessage(self,senderID,receiverID,msg):
        encoded = self.encodeMessage(msg)
        if self.noised:
            encoded = self.addNoise(encoded)
        
        pass

    def recieveMessage(self,senderID,receiverID):
        pass

    def decodeMessage(self):
        pass