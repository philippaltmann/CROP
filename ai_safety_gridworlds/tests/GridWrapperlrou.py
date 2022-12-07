import gym
from gym import spaces
from gym import Env
import numpy as np
class BasicWrapperlrou(gym.Env):
    def __init__(self, env):
        super(BasicWrapperlrou, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(4,))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
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
        newArray = np.empty(shape=(4,))

        newArray[0] = board[x-1][y]
        newArray[1] = board[x][y-1]
        newArray[2] = board[x+1][y]
        newArray[3] = board[x][y+1]
        return newArray


    def seed(self,seed):
        print(seed)

    def addToObservations(self,observation):
        if observation.flatten().tolist() not in self.boardObservations:
            self.boardObservations.append(observation.flatten().tolist())
