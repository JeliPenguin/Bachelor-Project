class ErrorDetector():
    def __init__(self) -> None:
        pass

    def stringify(self, encoded):
        encodedString = ""
        for b in encoded:
            encodedString += str(b)
        return encodedString
    
    def encode(self,msg):
        """
        Return numpy array of bits
        """
        pass

    def decode(self,msg):
        """
        Returns tuple (correct,msg)
        correct: Whether the decoded message passes the test
        msg: The original message without the detection code
        """
        pass