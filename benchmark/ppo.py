from typing import Union
from algorithm import TrainableAlgorithm
from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.ppo import PPO as StablePPO

class PPO(TrainableAlgorithm, StablePPO):
  """A Trainable extension to PPO"""
