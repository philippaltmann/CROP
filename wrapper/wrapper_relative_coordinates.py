import gym
from gym import spaces
from gym import Env
import numpy as np
"""
Gym-Wrapper for the Observation-Shaping Method 'relative coordinates'
"""
class Wrapper_relative_coordinates(gym.Env):
    def __init__(self, env):
        super(Wrapper_relative_coordinates, self).__init__()
        self.env = env
        spaces_dic = {
            'goal' : spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
            'wall1': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
            'wall2': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
            'lava1': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
            'lava2': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
        }
        dict_space = gym.spaces.Dict(spaces_dic)
        self.observation_space = dict_space
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
        reward = timestep.reward

        next_state = self.get_ents(timestep.observation['board'],timestep.reward)

        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']


        return next_state, reward, done, info

    def reset(self):
        """
        Resets the original environment
        """
        timestep = self.env.reset()
        next_state = self.get_ents(timestep.observation['board'], timestep.reward)

        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']

        return next_state



    def get_ents(self,observation,reward):
        """
        Gets the relative coordinates from the player to the next 2 walls, 2 lava and the goal
        
        :param board: original board
        :param reward: reward to check if the position of the agent is lava
        :return: dictionary with coordinates to the entities
        """
        goal_state =   (0, 0)
        player_state = (0, 0)
        wall_state_1 = (0, 0)
        wall_state_2 = (0, 0)
        lava_state_1 = (0, 0)
        lava_state_2 = (0, 0)

        Env_Y = len(observation)    # 7 in original
        Env_X = len(observation[0]) # 9 in original
 
        for i in range(Env_Y):
            for j in range(Env_X):
                if observation[i][j] == 2:
                    player_state = (j,i)

        nearest_lava_1 = 99
        nearest_lava_2 = 99
        nearest_wall_1 = 99
        nearest_wall_2 = 99

        for x in range(Env_Y):
            for y in range(Env_X):
                if observation[x][y] == 4:

                    distance = abs(player_state[0]-y) + abs(player_state[1]-x)

                    if(distance < nearest_lava_1):
                        lava_state_2 = lava_state_1
                        lava_state_1 = y - player_state[0], x - player_state[1]
                        nearest_lava_2 = nearest_lava_1
                        nearest_lava_1 = distance

                    elif (distance < nearest_lava_2):
                        lava_state_2 = y - player_state[0], x - player_state[1]
                        nearest_lava_2 = distance


                if observation[x][y] == 0:
                    distance = abs(player_state[0]-y) + abs(player_state[1]-x)
                    if(distance < nearest_wall_1):
                        wall_state_2 = wall_state_1
                        wall_state_1 = y - player_state[0], x - player_state[1]
                        nearest_wall_2 = nearest_wall_1
                        nearest_wall_1 = distance
                    elif (distance < nearest_wall_2):
                        wall_state_2 = y - player_state[0], x - player_state[1]
                        nearest_wall_2 = distance

                if observation[x][y] == 3:
                    goal_state = y - player_state[0], x - player_state[1]

        if(reward == -51):
            lava_state_2 = lava_state_1
            lava_state_1 = (0,0)

        obs_dic = {
            'goal': [goal_state[0],goal_state[1]],
            'wall1': [wall_state_1[0],wall_state_1[1]],
            'wall2': [wall_state_2[0],wall_state_2[1]],
            'lava1': [lava_state_1[0],lava_state_1[1]],
            'lava2': [lava_state_2[0],lava_state_2[1]],
        }

        return obs_dic







