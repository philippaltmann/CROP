import gym
from gym import spaces
from gym import Env
import numpy as np
class BasicWrapperDistance(gym.Env):
    def __init__(self, env):
        super(BasicWrapperDistance, self).__init__()
        self.env = env
        spaces_dic = {
            'grid': spaces.Box(low=0,high=4,shape=(5,5)),
            'distance_lava': spaces.Discrete(6),
        }
        dict_space = gym.spaces.Dict(spaces_dic)
        self.observation_space = dict_space
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.spec = env.observation_spec
        self.boardObservations = []


    def step(self, action):
        timestep = self.env.step(action)
        restrictedView = self.modifyBoard(timestep.observation['board'])
        reward = timestep.reward
        distance_lava = self.getDistance(restrictedView,reward)
        spaces_dic = {
            'grid': restrictedView,
            'distance_lava': distance_lava
        }
        next_state = spaces_dic
        self.addToObservations(restrictedView)

        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        #next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        reward = timestep.reward
        restrictedView = self.modifyBoard(timestep.observation['board'])
        distance_lava = self.getDistance(restrictedView,reward)
        spaces_dic = {
            'grid': restrictedView,
            'distance_lava': distance_lava
        }
        next_state = spaces_dic
        self.addToObservations(restrictedView)

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
            nearestLavaState = 0,0
        return nearestLava


    def seed(self,seed):
        print(seed)

    def addToObservations(self,observation):
        if observation.flatten().tolist() not in self.boardObservations:
            self.boardObservations.append(observation.flatten().tolist())
