EMPTY = "-"
TREAT = "$"

UP = 0
LEFT = 1
RIGHT = 2
DOWN = 3
STAY = 4
ACTIONSPACE = ["UP", "LEFT", "RIGHT", "DOWN", "STAY"]


def decodeAction(num: int):
    mapping = {
        0: (-1, 0),
        1: (0, -1),
        2: (0, 1),
        3: (1, 0),
        4: (0, 0)
    }
    return mapping[num]


def transition(initStateTuple, actionTuple,strict=False):
    if strict:
        return tuple(max(0,x + y) for x, y in zip(initStateTuple, actionTuple))
    return tuple(x + y for x, y in zip(initStateTuple, actionTuple))


GUIDEID = 0
startingScoutID = GUIDEID + 1
