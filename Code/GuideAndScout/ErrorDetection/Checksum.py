from ErrorDetection.ErrorDetector import ErrorDetector
import numpy as np

class Checksum(ErrorDetector):
    def __init__(self,k) -> None:
        super().__init__()
        self._k = k

    def calcDigitsum(self, binString: str):
        digitSum = 0
        for i in range(int(len(binString)/self._k)):
            digitSum += int(binString[self._k*i:self._k*(i+1)], 2)
        digitSum = bin(digitSum)[2:]
        return digitSum

    def encode(self, encoded):
        # Normal encoded length 168
        # Terminal state encoded length 88
        res = []
        encoded = self.stringify(encoded)
        digitSum = self.calcDigitsum(encoded)
        if(len(digitSum) > self._k):
            x = len(digitSum)-self._k
            digitSum = bin(int(digitSum[0:x], 2)+int(digitSum[x:], 2))[2:]
        if(len(digitSum) < self._k):
            digitSum = '0'*(self._k-len(digitSum))+digitSum

        for i in digitSum:
            if(i == '1'):
                res.append(0)
            else:
                res.append(1)

        return np.array(res, dtype=np.uint8)

    def decode(self, receivedMsg):
        # receivedMsg includes checksum as the final 8 bits
        strReceivedMsg = self.stringify(receivedMsg)
        digitSum = self.calcDigitsum(strReceivedMsg)
        # Adding the overflow bits
        if(len(digitSum) > self._k):
            x = len(digitSum)-self._k
            digitSum = bin(
                int(digitSum[0:x], 2)+int(digitSum[x:], 2))[2:]

        checksum = 0
        for i in digitSum:
            if(i == '1'):
                checksum += 0
            else:
                checksum += 1
        return (checksum == 0),receivedMsg[self._k:]