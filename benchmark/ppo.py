from typing import Union
from algorithm import TrainableAlgorithm
from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.ppo import PPO as StablePPO

class PPO(TrainableAlgorithm, StablePPO):
  """A Trainable extension to PPO"""
  # def __init__( self, *args, n_steps: int = 512,  **kwargs): super(PPO, self).__init__(*args, n_steps=n_steps, **kwargs)
  # def train(self) -> None:
  #   self.logger.record("rewards/environment", self.rollout_buffer.rewards.copy()) 
  #   super(PPO, self).train()

class PPO2(TrainableAlgorithm, StablePPO):
  """A Trainable extension to PPO \w Hyperparamters from https://arxiv.org/pdf/1912.01588.pdf"""

  def __init__( self, *args, gamma: float = 0.999, n_steps: int = 256, n_epochs: int = 3, 
    batch_size: int = 32, ent_coef: float = 0.01, learning_rate: Union[float, Schedule] = 3e-5, **kwargs):
      super(PPO2, self).__init__(*args, gamma=gamma, n_steps=n_steps, n_epochs=n_epochs, 
      batch_size=batch_size, ent_coef=ent_coef,  learning_rate=learning_rate, **kwargs)
  # def train(self) -> None:
  #   self.logger.record("rewards/environment", self.rollout_buffer.rewards.copy()) 
  #   super(PPO, self).train()
