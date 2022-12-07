import gym
from gym import spaces
from gym import Env
import numpy as np
"""
Gym-Wrapper for the Reward-Shaping Method 
"""
class Wrapper_reward(gym.Env):
    def __init__(self, env):
        super(Wrapper_reward, self).__init__()
        self.env = env
        self.observation_space = spaces.Box(low=0,high=4,shape=(7,9))
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.board_observations = []
        self.action_dict = {0: "u", 1: "d", 2: "l", 3: "r"}
        self.reward_coefficient = 3


    def step(self, action):
        """
        Execute a step in the original Environment and modify the reward.
        The observation is not modified
        :param board: action 
        :return: next_state, modified reward, done, info
        """
        timestep = self.env.step(action)
        next_state = timestep.observation['board']
        reward = self.shape_reward(timestep.observation['board'],timestep.reward)
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        info["action"] = self.action_dict[action]  # for Grid
        info["state"] = self.state  # for Grid state before the action
        self.state = self.get_state(timestep.observation['board'])  # for Grid
        if (done):
            info["finished_state"] = self.state
        return next_state, reward, done, info

    def reset(self):
        """
        Resets the original environment
        """
        self.state = (1, 1)
        timestep = self.env.reset()
        next_state =timestep.observation['board']

        reward = timestep.reward
        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']

        return next_state


    def get_distance(self,board,reward):
        """
        compute the distance from agent to the narest lava-tile
        :param board: original board
        :param reward: original reward
           :return: distanceX and distanceY from agent to nearest lava
        """
        nearest_lava = 10
        nearest_lava_state = (9,9)
        Env_Y = len(board)    # 7 in original
        Env_X = len(board[0]) # 9 in original

        playe_stateX, playe_stateY = self.get_state(board)
        for y in range(Env_Y):
            for x in range(Env_X):
                if board[y][x] == 4:
                    distance = abs(playe_stateX-x) + abs(playe_stateY-y)
                    if(distance < nearest_lava):
                        nearest_lava = distance
                        nearest_lava_state = y-playe_stateY,x-playe_stateX
        if(reward == -51):
            nearest_lava = 0
        return nearest_lava


    def add_to_observations(self,observation):
        """
        log function for evaluation
        """
        if observation.flatten().tolist() not in self.board_observations:
            self.board_observations.append(observation.flatten().tolist())

    def shape_reward(self,board,env_reward):
        """
        Modifies the Reward depending on the distance to the nearest lava. No Modification if the agent reached the Goal
        :param board: original board
        :param reward: original reward
           :return: shaped reward
        """
        if(env_reward)<0:
            distance = self.get_distance(board,env_reward)
            return -50/(self.reward_coefficient**distance)
        else:
            return env_reward

    def get_state(self,board):
        """
        Get agent coordinates
        """
        playe_state = (0,0)
        Env_Y = len(board)    # 7 in original
        Env_X = len(board[0]) # 9 in original

        for i in range(Env_Y):
            for j in range(Env_X):
                if board[i][j] == 2:
                    playe_state = (j,i)
        return playe_state
