from enum import Enum

class GridSymbol(Enum):
    EMPTY = "-"
    TREAT = "$"
    AGENT = "*"

class Action(Enum):
    UP = (-1,0)
    LEFT = (0,-1)
    RIGHT = (0,1)
    DOWN = (1,0)
    STAY = (0,0)

def encodeAction(num):
    mapping = {
        0:Action.UP,
        1:Action.LEFT,
        2:Action.RIGHT,
        3:Action.DOWN,
        4:Action.STAY
    }
    return mapping[num]