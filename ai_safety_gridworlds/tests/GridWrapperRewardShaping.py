import gym
from gym import spaces
from gym import Env
import numpy as np
class BasicWrapperRewardShaping(gym.Env):
    def __init__(self, env):
        super(BasicWrapperRewardShaping, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(7,9))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.spec = env.observation_spec
        self.boardObservations = []
        self.actionDict = {0: "u", 1: "d", 2: "l", 3: "r"}


    def step(self, action):
        timestep = self.env.step(action)
        #print(timestep.observation['board'])
        next_state = timestep.observation['board']
        #self.addToObservations(next_state)
        reward = self.shapeReward(self.modifyBoard(timestep.observation['board']),timestep.reward)
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']

        info["action"] = self.actionDict[action]  # for Grid
        info["state"] = self.state  # for Grid state before the action
        self.state = self.getState(timestep.observation['board'])  # for Grid

        if (done):
            info["finished_state"] = self.state
        #next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state, reward, done, info

    def reset(self):
        self.state = (1, 1)
        timestep = self.env.reset()
        next_state =timestep.observation['board']
        #self.addToObservations(next_state)
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

    def getDistance(self,board,reward):
        nearestLava = 5
        nearestLavaState = (3,3)
        playerStateX, playerStateY = 2,2
        for x in range(0,5):
            for y in range(0,5):
                if board[x][y] == 4:
                    distance = abs(playerStateX-x) + abs(playerStateY-y)
                    if(distance < nearestLava):
                        nearestLava = distance
                        nearestLavaState = y-playerStateY,x-playerStateX
        if(reward == -51):
            nearestLava = 0
        return nearestLava

    def seed(self,seed):
        print(seed)

    def addToObservations(self,observation):
        if observation.flatten().tolist() not in self.boardObservations:
            self.boardObservations.append(observation.flatten().tolist())

    def shapeReward(self,board,env_reward):

        if(env_reward)<0:
            distance = self.getDistance(board,env_reward)
            return -50/(0.5**distance)
        else:
            #print(env_reward)
            return env_reward*2

    def getState(self,board):
        playerState = (0,0)
        for i in range(7):
            for j in range(9):
                if board[i][j] == 2:
                    playerState = (j,i)
        return playerState
