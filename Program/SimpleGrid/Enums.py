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