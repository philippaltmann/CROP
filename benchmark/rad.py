"""Reinforcement Learning with Augmented Data (RAD) adapted from https://github.com/MishaLaskin/rad"""
import numpy as np
from ai_safety_gym.gym.wrapper import SafetyWrapper 

def RAD(fill=-1, cut_range=(0.04,0.24), crop=0.84) -> SafetyWrapper: #:Any[CropSpace,str] todo
  class RADWrapper(SafetyWrapper):
    def __init__(self, env, **kwargs):
      super(RADWrapper, self).__init__(env=env, **kwargs)
      self.fill = fill  # (CROP, CUT): (6,(0,2)) (9,(0,3)) | Maze7 Maze11
      self.cut_size = [(round(s*cut_range[0]), round(s*cut_range[1])) for s in self.observation_space.shape]
      self.crop_size = tuple([round(s*crop) for s in self.observation_space.shape])

    def step(self, action):
      state, reward, done, info = super(RADWrapper, self).step(action)
      return (self.augment(state), reward, done, info)

    def reset(self, game_art=None):
      state = super().reset(game_art=game_art)
      return self.augment(state)

    def augment(self, observation): 
      cropout = self.random_crop(observation)
      translate = self.random_translate(cropout)
      return self.random_cutout(translate)
          
    def random_cutout(self, obs):
      cut = [np.random.randint(*s) for s in self.cut_size]
      (wp,wc), (hp,hc) = [(np.random.randint(0, s - c + 1),c) for (c, s) in zip(cut, obs.shape[1:])]
      for o in obs: o[hp:hp + hc, wp:wp + wc] = self.fill
      return obs

    def random_translate(self, obs):
      size = self.observation_space.shape; 
      out = np.full((obs.shape[0],*size), self.fill, dtype=obs.dtype); assert size >= obs.shape[1:]
      (hp,h),(wp,w) = [(np.random.randint(0, s - o + 1), o) for (s, o) in zip(size, obs.shape[1:])]
      for i in range(obs.shape[0]): out[i, hp:hp + h, wp:wp + w] = obs
      return out

    def random_crop(self, obs):
      m = [(np.random.randint(0, o - s + 1, (obs.shape[0],))) for (o,s) in zip(obs.shape[1:], self.crop_size)]
      return np.array([o[h:h+self.crop_size[0], w:w+self.crop_size[1]] for (o, w, h) in zip(obs,*m)])

  return RADWrapper
