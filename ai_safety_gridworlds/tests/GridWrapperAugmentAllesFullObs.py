from enum import Enum

import gym
from ai_safety_gridworlds.tests.GridWrapperAugment import AugmentType
from gym import spaces
from gym import Env
import numpy as np
import random
class BasicWrapperAugmentAllesFullObs(gym.Env):
    def __init__(self, env, augmentGrad):
        super(BasicWrapperAugmentAllesFullObs, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=-1,high=4,shape=(7,9),dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.spec = env.observation_spec
        self.boardObservations = []
        self.augmentValue = 10
        self.augmentChoice = AugmentType(random.randint(1, 4))
        self.flipValue = random.randint(1, 2)
        self.rotateValue = random.randint(1, 3)
        self.state = (1, 1)
        self.actionDict = {0: "u", 1: "d", 2: "l", 3: "r"}
        self.augmentGrad = augmentGrad

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
        print("NextState:",next_state)

        return next_state, reward, done, info

    def reset(self):
        self.state = (1, 1)
        self.augmentValue = random.randint(1, 10)
        self.flipValue = random.randint(1, 2)
        self.rotateValue = random.randint(1, 3)
        self.augmentChoice = AugmentType(random.randint(1, 4))
        timestep = self.env.reset()

        next_state = self.augment(timestep)
        self.addToObservations(next_state)


        return next_state

    def augment(self,timestep):
        next_state = self.modifyBoard(timestep.observation['board'])
        self.augmentValue = random.randint(1, 10)
        if (self.augmentValue <= self.augmentGrad):
            # print("Augment: chance was ", self.augmentValue,"of", self.augmentGrad)
            self.augmentChoice = AugmentType(random.randint(1, 4))
            if AugmentType.FLIP == self.augmentChoice:
                # if i==1:
                # j = random.randint(1,2)
                if self.flipValue == 1:
                    #print("flip1")
                    next_state = self.flip(self.modifyBoard(timestep.observation['board']), "x")
                if self.flipValue == 2:
                    #print("flip2")
                    next_state = self.flip(self.modifyBoard(timestep.observation['board']), "y")
            if AugmentType.ROTATE == self.augmentChoice:
                # if i == 2:
                #j = random.randint(1,3)
                if self.rotateValue == 1:
                    #print("rotate1")
                    next_state = self.rotate(self.modifyBoard(timestep.observation['board']), 90)
                if self.rotateValue == 2:
                    #print("rotate2")
                    next_state = self.rotate(self.modifyBoard(timestep.observation['board']), 180)
                if self.rotateValue == 3:
                    #print("rotate3")
                    next_state = self.rotate(self.modifyBoard(timestep.observation['board']), 270)
            if AugmentType.CUT == self.augmentChoice:
                # if i == 3:
                #print("cut")
                next_state = self.cut(self.modifyBoard(timestep.observation['board']))
                # print("cut",next_state)

            if AugmentType.CROP == self.augmentChoice:
                # if i == 4:
                #print("crop")
                next_state = self.crop(timestep.observation['board'])
                # print("crop", next_state)


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
        return board

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
        new_x = random.randint(0, 7)
        new_y = random.randint(0, 9)

        playerState = (0, 0)
        for i in range(7):
            for j in range(9):
                if board[i][j] == 2:
                    playerState = (i, j)

        x, y = playerState
        newArray = np.zeros(shape=(7, 9))

        for i in range(-2,3):
            for j in range(-2,3):
                if x+i < 0 or x+i>6 or y+j < 0 or y+j>8:
                    newArray[i+2][j+2] = 0
                else:
                    newArray[i+2][j+2] = board[x+i][y+j]

        return newArray

    def modifyAction(self,action):
        modifiedAction = action
        if (self.augmentValue <= self.augmentGrad):
            if AugmentType.FLIP == self.augmentChoice:
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

            if AugmentType.ROTATE == self.augmentChoice:
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