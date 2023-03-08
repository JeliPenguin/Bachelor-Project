from ErrorDetection.ErrorDetector import ErrorDetector

class Hamming(ErrorDetector):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, msg):
        return super().encode(msg)
    
    def decode(self, msg):
        return super().decode(msg)