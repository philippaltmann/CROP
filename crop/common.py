from typing import Any, Tuple
import numpy as np
import gym
from ai_safety_gym import SafetyWrapper
from ai_safety_gym.environments.shared.safety_game import Actions

class Radius(gym.spaces.Box):
  def __init__(self, env, radius=(5,5)):
    self.mapping = env.unwrapped._env._value_mapping;
    values = list(self.mapping.values()); self.radius = radius
    super(Radius, self).__init__(low=values[0], high=values[-1], shape=radius, dtype=int)

  def crop(self, state): 
    state = np.squeeze(state, axis=0)
    pad = tuple(int((r-1)/2) for r in self.radius) 
    val = tuple(self.mapping['#'] for _ in self.radius)
    padded = np.pad(state, pad, 'constant', constant_values=val)
    pos = np.array(tuple(zip(*np.where(padded==2)))[0])
    s = [pos-pad,pos+pad+1]
    padded = padded[s[0][0]:s[1][0],s[0][1]:s[1][1]]
    return np.reshape(padded, self.shape)


class Action(gym.spaces.Box):
  def __init__(self, env):
    e = env.unwrapped._env
    values = list(e._value_mapping.values())
    self.agent = e._value_mapping['A']
    super(Action, self).__init__(low=values[0], high=values[-1], shape=(len(Actions.DEFAULT()),), dtype=int)

  def crop(self, state): 
    state = np.squeeze(state, axis=0)
    find = lambda value: list(zip(*np.where(state == value))) # or find(self.agent)
    a = np.array(find(self.agent))
    return np.array([state[tuple(p[0])] for p in Actions.iterate(a).values()])


class Object(gym.spaces.Box):
  def __init__(self, env, horizon=1): #TODO horizon > 1
    obs_shape = env.unwrapped._env.observation_spec()["board"].shape
    self.agent = env.unwrapped._env._value_mapping['A']
    self.values = [v for k,v in env.unwrapped._env._value_mapping.items() if k != 'A']
    self.horizon = horizon; self.shape = (len(self.values), horizon*2)
    high = np.array([list(obs_shape) for _ in range(horizon) for _ in self.values])
    super(Object, self).__init__(low=1, high=high, shape=self.shape, dtype=int)
    # high = np.array([[list(obs_shape) for _ in range(horizon)] for _ in self.values])
    # super(Object, self).__init__(low=1, high=high, shape=(len(self.values), horizon, 2), dtype=int)
  
  def crop(self, state): 
    state = np.squeeze(state, axis=0)
    find = lambda value: list(zip(*np.where(state == value))) # or find(self.agent)
    a = np.array(find(self.agent))
    def dist(v): 
      o = np.array(find(v) or a); d = a - o
      return d[np.argsort(np.sum(d**2,axis=1))][:self.horizon]
    return np.reshape(np.array([dist(v) for v in self.values]), self.shape)


class CropSpace(gym.spaces.Box): 
  def crop(self, state): raise NotImplementedError

def CROP(space, **args) -> Tuple[gym.Wrapper, str]: #:Any[CropSpace,str] todo
  class CROPWrapper(SafetyWrapper):
    def __init__(self, env, **kwargs):
      super(CROPWrapper, self).__init__(env=env, **kwargs)
      self.observation_space = eval(space)(env, **args)
    
    def step(self, action):
      state, reward, done, info = super(CROPWrapper, self).step(action)
      return (self.observation_space.crop(state), reward, done, info)

    def reset(self, game_art=None):
      state = super().reset(game_art=game_art)
      return self.observation_space.crop(state)

  return CROPWrapper, f'{space}CROP'
      
