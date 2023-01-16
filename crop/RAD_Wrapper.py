import gym
from gym import spaces
from gym import Env
import numpy as np
from data_augs import random_cutout, random_crop, random_translate
import random
"""
Gym-Wrapper for the implementation of RAD
Called by:

env = ai_safety_gridworlds.helpers.factory.get_environment_obj(environment_name, is_testing=False, level_choice=0)
wrapped_env = Wrapper_RAD(env)

"""
class Wrapper_RAD(gym.Env):
    def __init__(self, env):
        super(Wrapper_RAD, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=-1,high=5,shape=(8,8))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data


    def step(self, action):
        """
        Execute a step in the original Environment 
        :param action: action 
        :return: augmented observation
        """
        timestep = self.env.step(action)

        cropped_state = random_crop(timestep.observation['board'],out=6)
        translated_state = random_translate(cropped_state , size=8)
        cutted_state = random_cutout(translated_state)

        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']

        return cutted_state, reward, done, info

    def reset(self):
        """
        Resets the original environment
        """
        timestep = self.env.reset()

        cropped_state = random_crop(timestep.observation['board'],out=6)
        translated_state = random_translate(cropped_state , size=8)
        cutted_state = random_cutout(translated_state)

        return cutted_state