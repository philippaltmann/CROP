import gym
from gym import spaces
from gym import Env
import numpy as np
class BasicWrapper3x3(gym.Env):
    def __init__(self, env):
        super(BasicWrapper3x3, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(3,3))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.spec = env.observation_spec
        self.boardObservations = []


    def step(self, action):
        timestep = self.env.step(action)
        #print(timestep.observation['board'])
        next_state = self.modifyBoard(timestep.observation['board'])
        self.addToObservations(next_state)
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        #next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        next_state = self.modifyBoard(timestep.observation['board'])
        self.addToObservations(next_state)
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        # next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state

    def modifyBoard(self, board):
        playerState = (0,0)
        for i in range(7):
            for j in range(9):
                if board[i][j] == 2:
                    playerState = (i,j)

        x,y = playerState
        newArray = np.empty(shape=(3,3))
        for i in range(-1,2):
            for j in range(-1,2):
                if x+i < 0 or x+i>6 or y+j < 0 or y+j>8:
                    newArray[i+1][j+1] = 0
                else:
                    newArray[i+1][j+1] = board[x+i][y+j]
        return newArray


    def seed(self,seed):
        print(seed)

    def addToObservations(self,observation):
        if observation.flatten().tolist() not in self.boardObservations:
            self.boardObservations.append(observation.flatten().tolist())
