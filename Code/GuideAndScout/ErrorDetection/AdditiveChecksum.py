from ErrorDetection.ErrorDetector import ErrorDetector
import numpy as np

class AdditiveChecksum(ErrorDetector):
    def __init__(self,k=8) -> None:
        super().__init__()
        self._k = k

    def calcDigitsum(self, binString: str):
        digitSum = 0
        for i in range(int(len(binString)/self._k)):
            digitSum += int(binString[self._k*i:self._k*(i+1)], 2)
        digitSum = bin(digitSum)[2:]
        return digitSum
    
    def handleOverflow(self,digitSum):
        x = len(digitSum)-self._k
        newdigitSum = bin(int(digitSum[0:x], 2)+int(digitSum[x:], 2))[2:]
        return newdigitSum

    def encode(self, encoded):
        # Normal encoded length 168
        # Terminal state encoded length 88
        # Divisible by 8
        res = []
        encoded = self.stringify(encoded)
        digitSum = self.calcDigitsum(encoded)
        if(len(digitSum) > self._k):
            digitSum = self.handleOverflow(digitSum)
        if(len(digitSum) < self._k):
            digitSum = '0'*(self._k-len(digitSum))+digitSum

        for i in digitSum:
            if(i == '1'):
                res.append(0)
            else:
                res.append(1)

        return np.array(res, dtype=np.uint8)

    def decode(self, receivedMsg):
        strReceivedMsg = self.stringify(receivedMsg)
        digitSum = self.calcDigitsum(strReceivedMsg)
        # Adding the overflow bits
        if(len(digitSum) > self._k):
            digitSum = self.handleOverflow(digitSum)
        for i in digitSum:
            if(i == '1'):
                continue
            else:
                return False,receivedMsg[self._k:]
        return True,receivedMsg[self._k:]