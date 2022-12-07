import gym
from gym import spaces
from gym import Env
import numpy as np
"""
Gym-Wrapper for the Observation-Shaping Method '3x3'
"""
class Wrapper_3x3(gym.Env):
    def __init__(self, env):
        super(Wrapper_3x3, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(3,3))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.board_observations = []


    def step(self, action):
        """
        Executes a step in the original Environment and modifies the observation
        :param board: action 
        :return: modified observation
        """
        timestep = self.env.step(action)
        next_state = self.modify_board(timestep.observation['board'])
        self.add_to_observations(next_state)
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']

        return next_state, reward, done, info

    def reset(self):
        """
        Resets the original environment
        """
        timestep = self.env.reset()
        next_state = self.modify_board(timestep.observation['board'])
        self.add_to_observations(next_state)
        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        return next_state

    def modify_board(self, board):
        """
        Transform original board to a 3x3 board with the Agent in the center.
        
        :param board: original board
        :return: new cutout 3x3 board
        """
        Env_Y = len(board)    # 7 in original
        Env_X = len(board[0]) # 9 in original

        player_state = (0,0)
        for i in range(Env_Y):
            for j in range(Env_X):
                if board[i][j] == 2:
                    player_state = (i,j)

        x,y = player_state
        new_array = np.empty(shape=(3,3))
        for i in range(-1,2):
            for j in range(-1,2):
                if x+i < 0 or x+i > Env_X-1 or y+j < 0 or y+j > Env_X-1:
                    new_array[i+1][j+1] = 0
                else:
                    new_array[i+1][j+1] = board[x+i][y+j]
        return new_array

    def add_to_observations(self,observation):
        """
        log function for evaluation
        """
        if observation.flatten().tolist() not in self.board_observations:
            self.board_observations.append(observation.flatten().tolist())
