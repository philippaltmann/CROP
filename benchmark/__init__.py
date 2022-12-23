# from .a2c import A2C
# from .dqn import DQN
from .ppo import PPO, PPO2
from ai_safety_gym import FullyObservable
# from ai_safety_gym import FO
# TODO: SAC, RAD

ALGS = ['PPO', 'PPO2', 'FullyObservable']