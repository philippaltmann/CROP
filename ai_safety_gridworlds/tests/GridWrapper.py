import gym
from gym import spaces
from gym import Env
import numpy as np
class BasicWrapper(gym.Env):
    def __init__(self, env):
        super(BasicWrapper, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(7,9))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.state = (1,1)
        self.actionDict = { 0 : "u" , 1 : "d" ,2: "l" ,3 : "r"}

    def getState(self,board):
        playerState = (0,0)
        for i in range(7):
            for j in range(9):
                if board[i][j] == 2:
                    playerState = (j,i)
        return playerState

    def normalize_reward(self,reward):
        return reward/50

    def step(self, action):

        timestep = self.env.step(action)
        next_state = timestep.observation['board']
        #reward = self.normalize_reward(timestep.reward)
        reward = timestep.reward
        done = timestep.step_type == 2



        info = timestep.observation['extra_observations']


        info["action"] = self.actionDict[int(action)]
        info["state"] = self.state
        self.state = self.getState(next_state)
        if(done):
            info["finished_state"] = self.state


        #next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state, reward, done, info

    def reset(self):
        self.state = (1,1)
        timestep = self.env.reset()
        next_state = timestep.observation['board']
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        # next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state


    def seed(self,seed):
        print(seed)