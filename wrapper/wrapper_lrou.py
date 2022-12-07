import gym
from gym import spaces
from gym import Env
import numpy as np
"""
Gym-Wrapper for the Observation-Shaping Method 'lrou'
"""
class Wrapper_lrou(gym.Env):
    def __init__(self, env):
        super(Wrapper_lrou, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(4,))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.board_observations = []


    def step(self, action):
        """
        Execute a step in the original Environment and modifies the observation
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
        Transform original board to an array with only the information of the fields right next to the agent
        
        :param board: original board
        :return: new cutout lrou board
        """
        playe_state = (0,0)
        Env_Y = len(board)    # 7 in original
        Env_X = len(board[0]) # 9 in original

        for i in range(Env_Y):
            for j in range(Env_X):
                if board[i][j] == 2:
                    playe_state = (i,j)

        x,y = playe_state
        new_array = np.empty(shape=(4,))

        new_array[0] = board[x-1][y]
        new_array[1] = board[x][y-1]
        new_array[2] = board[x+1][y]
        new_array[3] = board[x][y+1]
        return new_array

    def add_to_observations(self,observation):
        """
        log function for evaluation
        """
        if observation.flatten().tolist() not in self.board_observations:
            self.board_observations.append(observation.flatten().tolist())
