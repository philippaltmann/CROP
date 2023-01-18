from .a2c import A2C
from .ppo import PPO
from .rad import RAD
from ai_safety_gym import SafetyWrapper

ALGS = ['A2C', 'PPO', 'RAD', 'Full']

def Observation(shape):
  
  if shape == 'Full': return SafetyWrapper
  assert False, f"{shape} Observation Not Implemented"
