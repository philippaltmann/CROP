import gym
from gym import spaces
from gym import Env
import numpy as np
import random
class BasicWrapperRestrictedView(gym.Env):
    def __init__(self, env):
        super(BasicWrapperRestrictedView, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(5,5),dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.spec = env.observation_spec
        self.boardObservations = []


    def step(self, action):
        timestep = self.env.step(action)
        #print(timestep.observation['board'])
        next_state = self.modifyBoard(timestep.observation['board'])

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
        newArray = np.empty(shape=(5,5))
        for i in range(-2,3):
            for j in range(-2,3):
                if x+i < 0 or x+i>6 or y+j < 0 or y+j>8:
                    newArray[i+2][j+2] = 0
                else:
                    newArray[i+2][j+2] = board[x+i][y+j]
        return newArray

    def rotate(self,board,angle):
        returnBoard = board
        if (angle == 90):
            returnBoard = np.rot90( board)
        if (angle == 180):
            returnBoard = np.rot90( board,2)
        if (angle == 270):
            returnBoard = np.rot90(board, 3)
        return returnBoard

    def flip(self,board,axes):
        returnBoard = board
        if (axes == "y"):
            returnBoard = np.fliplr( board)
        if (axes == "x"):
            returnBoard = np.flipud( board)

        return returnBoard


    def seed(self,seed):
        print(seed)

    def addToObservations(self,observation):
        if observation.flatten().tolist() not in self.boardObservations:
            self.boardObservations.append(observation.flatten().tolist())
