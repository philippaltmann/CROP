from enum import Enum

import gym
from gym import spaces
from gym import Env
import numpy as np
import random
class BasicWrapperAugment(gym.Env):
    def __init__(self, env, augmentGrad,augmentList):
        super(BasicWrapperAugment, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=-1,high=4,shape=(5,5),dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.spec = env.observation_spec
        self.boardObservations = []
        self.augmentValue = random.randint(1, 10)
        self.flipValue = random.randint(1, 2)
        self.rotateValue = random.randint(1, 3)
        self.state = (1, 1)
        self.actionDict = {0: "u", 1: "d", 2: "l", 3: "r"}
        self.augmentGrad = augmentGrad
        self.augmentList = augmentList

    def getState(self,board):
        playerState = (0,0)
        for i in range(7):
            for j in range(9):
                if board[i][j] == 2:
                    playerState = (j,i)
        return playerState

    def step(self, action):
        timestep = self.env.step(self.modifyAction(action))
        #print(timestep.observation['board'])

        next_state = self.augment(timestep)


        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        info["action"] = self.actionDict[self.modifyAction(action)]
        info["state"] = self.state
        self.state = self.getState(timestep.observation['board'])
        #next_state, reward, done, info = self.env.step(action)
        #print("NextState:",next_state)
        return next_state, reward, done, info

    def augment(self, timestep):
        next_state = self.modifyBoard(timestep.observation['board'])

        self.augmentValue = random.randint(1, 10)
        if (self.augmentValue <= self.augmentGrad):
            #print("Augment: chance was ", self.augmentValue,"of", self.augmentGrad)
            # i = random.randint(1,2)
            if AugmentType.FLIP in self.augmentList:

                # if i==1:
                # j = random.randint(1,2)
                if self.flipValue == 1:
                # print("flip1")
                    next_state = self.flip(self.modifyBoard(timestep.observation['board']),"x")
                if self.flipValue == 2:
                    #print("flip2")
                    next_state = self.flip(self.modifyBoard(timestep.observation['board']),"y")
            if AugmentType.ROTATE in self.augmentList:
                # if i == 2:
                # j = random.randint(1,3)
                if self.rotateValue == 1:
                    #print("rotate1")
                    next_state = self.rotate(self.modifyBoard(timestep.observation['board']), 90)
                    #print(next_state)
                if self.rotateValue == 2:
                    #print("rotate2")
                    next_state = self.rotate(self.modifyBoard(timestep.observation['board']), 180)
                    #print(next_state)
                if self.rotateValue == 3:
                    #print("rotate3")
                    next_state = self.rotate(self.modifyBoard(timestep.observation['board']), 270)
                    #print(next_state)
            if AugmentType.CUT in self.augmentList:
            # if i == 1:
            # print("cut")
                next_state = self.cut(self.modifyBoard(timestep.observation['board']))
            # print("cut",next_state)

            if AugmentType.CROP in self.augmentList:
            # if i == 2:
            # print("crop")
                next_state = self.crop(timestep.observation['board'])
            # print("crop", next_state)
        #else:
            #print("No Augment: chance was ", self.augmentValue,"of", self.augmentGrad)
            #print(next_state)

        return next_state

    def reset(self):
        self.state = (1, 1)
        self.augmentValue = random.randint(1,10)
        self.flipValue = random.randint(1, 2)
        self.rotateValue = random.randint(1, 3)
        timestep = self.env.reset()
        next_state = self.augment(timestep)
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

    def cut(self,board):
        returnBoard = board
        x1 = random.randint(0,4)
        y1 = random.randint(0, 4)
        x2 = random.randint(0,4)
        y2 = random.randint(0, 4)
        x3 = random.randint(0,4)
        y3 = random.randint(0, 4)
        x4 = random.randint(0,4)
        y4 = random.randint(0, 4)

        returnBoard[x1][y1] = -1
        returnBoard[x2][y2] = -1
        returnBoard[x3][y3] = -1
        returnBoard[x4][y4] = -1

        returnBoard[2][2] = 2

        return returnBoard


    def seed(self,seed):
        print(seed)

    def addToObservations(self,observation):
        if observation.flatten().tolist() not in self.boardObservations:
            self.boardObservations.append(observation.flatten().tolist())

    def crop(self, board):
        new_x = random.randint(0, 4)
        new_y = random.randint(0, 4)

        playerState = (0, 0)
        for i in range(7):
            for j in range(9):
                if board[i][j] == 2:
                    playerState = (i, j)

        x, y = playerState
        newArray = np.empty(shape=(5, 5))
        for i in range(0, 5):
            for j in range(0, 5):
                if i == new_x and j == new_y:
                    newArray[i][j] = 2
                elif x-new_x+i < 0 or x-new_x+i > 6 or y - new_y+j < 0 or y - new_y+j > 8:
                    newArray[i][j] = 0
                else:
                    newArray[i][j] = board[x-new_x+i][y - new_y+j]
        return newArray

    def modifyAction(self,action):
        modifiedAction = action
        if (self.augmentValue <= self.augmentGrad):
            if AugmentType.FLIP in self.augmentList:
                if self.flipValue == 1:
                    if action == 0:
                        modifiedAction = 1
                    if action == 1:
                        modifiedAction = 0
                if self.flipValue == 2:
                    if action == 2:
                        modifiedAction = 3
                    if action == 3:
                        modifiedAction = 2

            if AugmentType.ROTATE in self.augmentList:
                if self.rotateValue == 1:
                    if action == 0:
                        modifiedAction = 3
                    if action == 1:
                        modifiedAction = 2
                    if action == 2:
                        modifiedAction = 0
                    if action == 3:
                        modifiedAction = 1

                if self.rotateValue == 2:
                    if action == 0:
                        modifiedAction = 1
                    if action == 1:
                        modifiedAction = 0
                    if action == 2:
                        modifiedAction = 3
                    if action == 3:
                        modifiedAction = 2

                if self.rotateValue == 3:
                    if action == 0:
                        modifiedAction = 2
                    if action == 1:
                        modifiedAction = 3
                    if action == 2:
                        modifiedAction = 1
                    if action == 3:
                        modifiedAction = 0


        return modifiedAction


class AugmentType(Enum):
    CROP = 1
    CUT = 2
    FLIP = 3
    ROTATE = 4