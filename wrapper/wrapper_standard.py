import gym
from gym import spaces
from gym import Env
import numpy as np
"""
Gym-Wrapper for the Observation-Shaping Method '5x5'
"""
class Wrapper_standard(gym.Env):
    def __init__(self, env):
        super(Wrapper_standard, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(7,9))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.state = (1,1)
        self.action_dict = { 0 : "u" , 1 : "d" ,2: "l" ,3 : "r"}

    def get_state(self,board):
        """
        Gets the agents coordinates, only used for evaluation
        :param board: original environment board
           :return: agents coordinates
        """
        Env_Y = len(board)    # 7 in original
        Env_X = len(board[0]) # 9 in original

        playe_state = (0,0)
        for i in range(Env_Y):
            for j in range(Env_X):
                if board[i][j] == 2:
                    playe_state = (j,i)
        return playe_state


    def step(self, action):
        """
        Execute a step in the original Environment 
        :param board: action 
           :return: not modified observation
        """
        timestep = self.env.step(action)
        next_state = timestep.observation['board']
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        info["action"] = self.action_dict[int(action)]
        info["state"] = self.state
        self.state = self.get_state(next_state)
        if(done):
            info["finished_state"] = self.state
        return next_state, reward, done, info

    def reset(self):
        """
        Resets the original environment
        """
        self.state = (1,1)
        timestep = self.env.reset()
        next_state = timestep.observation['board']
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        return next_state