import gym
from gym import spaces
from gym import Env
import numpy as np
class BasicWrapperDynamic(gym.Env):
    def __init__(self, env):
        super(BasicWrapperDynamic, self).__init__()
        self.env = env
        spaces_dic = {
            'goal': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
            'wall1': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
            'wall2': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
            'lava1': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
            'lava2': spaces.Box(low=np.array([-6, -8]), high=np.array([6, 8]), dtype=np.int32,shape=(2,)),
        }
        dict_space = gym.spaces.Dict(spaces_dic)
        self.observation_space = dict_space
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.spec = env.observation_spec
        self.boardObservations = []


    def step(self, action):
        timestep = self.env.step(action)
        reward = timestep.reward

        next_state = self.getEnts(timestep.observation['board'],timestep.reward)

        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        #next_state, reward, done, info = self.env.step(action)
        # modify ...

        return next_state, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        next_state = self.getEnts(timestep.observation['board'], timestep.reward)

        done = timestep.step_type == 2
        info = timestep.observation['extra_observations']
        # next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state


    def seed(self,seed):
        print(seed)


    #0=wall,1=floor,2=lava,3=goal
    def getEnts(self,observation,reward):
        goal_state = (0,0)
        player_state = (0,0)
        wall_state_1 = (0, 0)
        wall_state_2 = (0, 0)
        lava_state_1 = (0, 0)
        lava_state_2 = (0, 0)

        for i in range(7):
            for j in range(9):
                if observation[i][j] == 2:
                    player_state = (j,i)

        nearestLava1 = 99
        nearestLava2 = 99
        nearestWall1 = 99
        nearestWall2 = 99

        for x in range(7):
            for y in range(9):
                if observation[x][y] == 4:

                    distance = abs(player_state[0]-y) + abs(player_state[1]-x)

                    if(distance < nearestLava1):
                        lava_state_2 = lava_state_1
                        lava_state_1 = y - player_state[0], x - player_state[1]
                        nearestLava2 = nearestLava1
                        nearestLava1 = distance

                    elif (distance < nearestLava2):
                        lava_state_2 = y - player_state[0], x - player_state[1]
                        nearestLava2 = distance


                if observation[x][y] == 0:
                    distance = abs(player_state[0]-y) + abs(player_state[1]-x)
                    if(distance < nearestWall1):
                        wall_state_2 = wall_state_1
                        wall_state_1 = y - player_state[0], x - player_state[1]
                        nearestWall2 = nearestWall1
                        nearestWall1 = distance
                    elif (distance < nearestWall2):
                        wall_state_2 = y - player_state[0], x - player_state[1]
                        nearestWall2 = distance

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







