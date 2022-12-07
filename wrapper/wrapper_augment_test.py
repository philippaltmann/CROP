import gym
from gym import spaces
from gym import Env
import numpy as np
"""
Gym-Wrapper for the Data Augmentation Method (only for testing)
No Data Augmentations will be executed here, but this wrapper has the same obersvation-space like 
agents trained with data augmentation
"""
class Wrapper_augment_test(gym.Env):
    def __init__(self, env):
        super(Wrapper_augment_test, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=-1,high=4,shape=(5,5))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.board_observations = []
        self.action_dict = {0: "u", 1: "d", 2: "l", 3: "r"}


    def step(self, action):
        """
        Execute a step in the original Environment. 
        :param board: action 
        :return: observation
        """
        timestep = self.env.step(action)
        next_state = self.modify_board(timestep.observation['board'])
        self.add_to_observations(next_state)
        reward = timestep.reward
        done = timestep.step_type == 2

        info = timestep.observation['extra_observations']
        info["action"] = self.action_dict[int(action)]            
        info["state"] = self.state                                  
        self.state = self.get_state(timestep.observation['board'])  

        if(done):
            info["finished_state"] = self.state

        return next_state, reward, done, info

    def reset(self):
        self.state = (1, 1)
        timestep = self.env.reset()
        next_state = self.modify_board(timestep.observation['board'])
        self.add_to_observations(next_state)
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']

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


    def seed(self,seed):
        print(seed)

    def add_to_observations(self,observation):
        if observation.flatten().tolist() not in self.board_observations:
            self.board_observations.append(observation.flatten().tolist())

    def get_state(self,board):
        playe_state = (0,0)
        Env_Y = len(board)    # 7 in original
        Env_X = len(board[0]) # 9 in original

        for i in range(Env_Y):
            for j in range(Env_X):
                if board[i][j] == 2:
                    playe_state = (j,i)
        return playe_state
