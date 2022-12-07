from enum import Enum
class RLAlgo(Enum):
	PPO = 0
	DQN = 1

class Wrapper(Enum):
    basic = 0
    reward_metrik = 1
    lrou = 2
    threexthree = 3
    fivexfive = 4
    relative = 5
    augment_training = 6
    augment_testing = 7

class AugmentType(Enum):
    CROP = 1
    CUT = 2
    FLIP = 3
    ROTATE = 4