from ErrorDetection.ErrorDetector import ErrorDetector
from crc import Calculator,Crc8
import numpy as np

class CRC(ErrorDetector):
    def __init__(self) -> None:
        super().__init__()
        self._calculator = Calculator(Crc8.CCITT)
        # Encoded into 9 bits binary code since using CRC8
        self._codeLength = 9

    def encode(self, msg):
        binEncode = bytes(msg)
        code = self._calculator.checksum(binEncode)
        return np.array(list(f'{code:0{self._codeLength}b}'),dtype=np.uint8)
    
    def binToDec(self,binArray):
        dec = 0
        reversedArr = binArray[::-1]
        for i in range(len(reversedArr)):
            dec += (2**i)*reversedArr[i]
        return dec
    
    def decode(self, msg):
        code = self.binToDec(msg[:self._codeLength])
        content = msg[self._codeLength:]
        correct = self._calculator.verify(content,code)
        return correct,content