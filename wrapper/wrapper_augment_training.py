from enum import Enum

import gym
from scripts.Enums import *
from gym import spaces
from gym import Env
import numpy as np
import random
"""
Gym-Wrapper for the Data Augmentation Method (for training)
"""
class Wrapper_augment_training(gym.Env):
    def __init__(self, env, augment_chance):
        super(Wrapper_augment_training, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=-1,high=4,shape=(5,5),dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.board_observations = []	#for logging
        self.augment_value = 10
        self.augment_choice = AugmentType(random.randint(1, 4))
        self.flip_value = random.randint(1, 2)
        self.rotate_value = random.randint(1, 3)
        self.state = (1, 1)
        self.action_dict = {0: "u", 1: "d", 2: "l", 3: "r"}
        self.augment_chance = augment_chance

    def get_state(self,board):
        """
        Returns the coordinats of the agent
        :param board: original board
        :return: coordinats of the agent
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
        Execute a step in the original Environment. 
        If last observation was modified by rotate or flip, the action will be modified first.
        :param board: action from agents Point-Of-View
           :return: possibly augmented observation
        """
        timestep = self.env.step(self.modify_action(action))

        next_state = self.augment(timestep)
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        info["action"] = self.action_dict[self.modify_action(action)]
        info["state"] = self.state
        self.state = self.get_state(timestep.observation['board'])

        return next_state, reward, done, info



    def augment(self,timestep):
        """
        roll a value to augment this timestep. If the value is below a threshold, then augment this timestep with 
        a random augmentation method. Flip and rotate variants stay the same over the whole episode.
        :param board: timestep (board,reward,done,info)
        :return: augmented observation
        """
        next_state = self.modify_board(timestep.observation['board'])
        self.augment_value = random.randint(1, 10)
        if (self.augment_value <= self.augment_chance):

            self.augment_choice = AugmentType(random.randint(1, 4))
			
            if AugmentType.FLIP == self.augment_choice:
                # j = random.randint(1,2)
                if self.flip_value == 1:
                    next_state = self.flip(self.modify_board(timestep.observation['board']), "x")
                if self.flip_value == 2:
                    next_state = self.flip(self.modify_board(timestep.observation['board']), "y")
					
            if AugmentType.ROTATE == self.augment_choice:
                #j = random.randint(1,3)
                if self.rotate_value == 1:
                    next_state = self.rotate(self.modify_board(timestep.observation['board']), 90)
                if self.rotate_value == 2:
                    next_state = self.rotate(self.modify_board(timestep.observation['board']), 180)
                if self.rotate_value == 3:
                    next_state = self.rotate(self.modify_board(timestep.observation['board']), 270)
            if AugmentType.CUT == self.augment_choice:
                next_state = self.cut(self.modify_board(timestep.observation['board']))

            if AugmentType.CROP == self.augment_choice:
                next_state = self.crop(timestep.observation['board'])


        return next_state
	
    def reset(self):
        """
        reset the environment. Roll new values to choose the flip or rotate variant
        """
        self.state = (1, 1)
        self.augment_value = random.randint(1, 10)
        self.flip_value = random.randint(1, 2)
        self.rotate_value = random.randint(1, 3)
        self.augment_choice = AugmentType(random.randint(1, 4))
        timestep = self.env.reset()

        next_state = self.augment(timestep)
        self.add_to_observations(next_state)


        return next_state

		
    def modify_board(self, board):
        """
        shrink the original observation to a 5x5 board
        :param board: original board
        :return: 5x5 board
        """
        Env_Y = len(board)    # 7 in original
        Env_X = len(board[0]) # 9 in original

        playe_state = (0,0)
        for i in range(Env_Y):
            for j in range(Env_X):
                if board[i][j] == 2:
                    playe_state = (i,j)

        x,y = playe_state
        new_array = np.empty(shape=(5,5))
        for i in range(-2,3):
            for j in range(-2,3):
                if x+i < 0 or x+i> Env_Y-1 or y+j < 0 or y+j> Env_X-1:
                    new_array[i+2][j+2] = 0
                else:
                    new_array[i+2][j+2] = board[x+i][y+j]
        return new_array

    def rotate(self,board,angle):
        """
        rotate the original board 90,180,270 degrees
        :param board: original board
           :return: new rotated board
        """
        return_board = board
        if (angle == 90):
            return_board = np.rot90( board)
        if (angle == 180):
            return_board = np.rot90( board,2)
        if (angle == 270):
            return_board = np.rot90(board, 3)
        return return_board

    def flip(self,board,axes):
        """
        flip the original board either on x or y axis
        :param board: original board
           :return: new flipped board
        """
        return_board = board
        if (axes == "y"):
            return_board = np.fliplr( board)
        if (axes == "x"):
            return_board = np.flipud( board)

        return return_board

    def cut(self,board):
        """
        randomly conceal parts of the board
        :param board: original board
           :return: new cutout board
        """
        return_board = board
        x1 = random.randint(0,4)
        y1 = random.randint(0, 4)
        x2 = random.randint(0,4)
        y2 = random.randint(0, 4)
        x3 = random.randint(0,4)
        y3 = random.randint(0, 4)
        x4 = random.randint(0,4)
        y4 = random.randint(0, 4)

        return_board[x1][y1] = -1
        return_board[x2][y2] = -1
        return_board[x3][y3] = -1
        return_board[x4][y4] = -1

        return_board[2][2] = 2

        return return_board


    def add_to_observations(self,observation):
        """
        log function for evaluation
        """
        if observation.flatten().tolist() not in self.board_observations:
            self.board_observations.append(observation.flatten().tolist())

    def crop(self, board):
        """
        random crop the original board
        :param board: original board
           :return: new cropped board
        """
        new_x = random.randint(0, 4)
        new_y = random.randint(0, 4)
        Env_Y = len(board)    # 7 in original
        Env_X = len(board[0]) # 9 in original

        playe_state = (0, 0)
        for i in range(Env_Y):
            for j in range(Env_X):
                if board[i][j] == 2:
                    playe_state = (i, j)

        x, y = playe_state
        new_array = np.empty(shape=(5, 5))
        for i in range(0, 5):
            for j in range(0, 5):
                if i == new_x and j == new_y:
                    new_array[i][j] = 2
                elif x-new_x+i < 0 or x-new_x+i > Env_Y-1 or y - new_y+j < 0 or y - new_y+j > Env_X -1:
                    new_array[i][j] = 0
                else:
                    new_array[i][j] = board[x-new_x+i][y - new_y+j]
        return new_array

    def modify_action(self,action):
        """
        modify the action matching the last observation-augmentation if the last 
        observation was augmented and if the last observation was augmented with 'flip' or 'rotate'
        :param action: action Point-Of-View from the agent
        :return: action Point-Of-View from the original environment
        """
        modified_action = action
        if (self.augment_value <= self.augment_chance):
            if AugmentType.FLIP == self.augment_choice:
                if self.flip_value == 1:
                    if action == 0:
                        modified_action = 1
                    if action == 1:
                        modified_action = 0
                if self.flip_value == 2:
                    if action == 2:
                        modified_action = 3
                    if action == 3:
                        modified_action = 2

            if AugmentType.ROTATE == self.augment_choice:
                if self.rotate_value == 1:
                    if action == 0:
                        modified_action = 3
                    if action == 1:
                        modified_action = 2
                    if action == 2:
                        modified_action = 0
                    if action == 3:
                        modified_action = 1

                if self.rotate_value == 2:
                    if action == 0:
                        modified_action = 1
                    if action == 1:
                        modified_action = 0
                    if action == 2:
                        modified_action = 3
                    if action == 3:
                        modified_action = 2

                if self.rotate_value == 3:
                    if action == 0:
                        modified_action = 2
                    if action == 1:
                        modified_action = 3
                    if action == 2:
                        modified_action = 1
                    if action == 3:
                        modified_action = 0


        return modified_action