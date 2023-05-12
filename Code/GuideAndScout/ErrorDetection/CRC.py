from ErrorDetection.ErrorDetector import ErrorDetector
from crc import Calculator,Crc8
import numpy as np

class CRC(ErrorDetector):
    """Uses the crc library from https://pypi.org/project/crc/"""
    def __init__(self) -> None:
        super().__init__()
        self._calculator = Calculator(Crc8.CCITT)
        # Encoded into 9 bits binary code since using CRC8
        self._codeLength = 9

    def encode(self, msg):
        binEncode = bytes(self.stringify(msg),"utf8")
        # print("Encoding Message: ",binEncode,type(binEncode))
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
        binEncode = bytes(self.stringify(content),"utf8")
        # print("Code recieved: ",msg[:self._codeLength])
        # print("Recived: ",binEncode,type(binEncode))
        correct = self._calculator.verify(binEncode,code)
        return correct,content