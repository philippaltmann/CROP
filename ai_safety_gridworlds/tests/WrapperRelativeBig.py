import gym
from gym import spaces
from gym import Env
import numpy as np
class WrapperRelative(gym.Env):
    def __init__(self, env):
        super(WrapperRelative, self).__init__()
        self.env = env
        spaces_dic = {
            'entity1': spaces.Box(low=np.array([0,-2, -2]), high=np.array([3,2, 2]), dtype=np.int32),
            'entity2': spaces.Box(low=np.array([0,-2, -2]), high=np.array([3,2, 2]), dtype=np.int32),
            'entity3': spaces.Box(low=np.array([0,-2, -2]), high=np.array([3,2, 2]), dtype=np.int32),
            'entity4': spaces.Box(low=np.array([0,-2, -2]), high=np.array([3,2, 2]), dtype=np.int32),
            'entity5': spaces.Box(low=np.array([0,-2, -2]), high=np.array([3,2, 2]), dtype=np.int32),
            'entity6': spaces.Box(low=np.array([0,-2, -2]), high=np.array([3,2, 2]), dtype=np.int32),
        }
        dict_space = gym.spaces.Dict(spaces_dic)
        self.observation_space = dict_space
        self.action_space = spaces.Discrete(4)
        self.metadata = env.environment_data
        self.spec = env.observation_spec
        self.boardObservations = []



    def step(self, action):
        timestep = self.env.step(action)
        board = timestep.observation['board']
        info = timestep.observation['extra_observations']
        reward = timestep.reward
        distance_lava = self.getDistanceLava(board,reward)
        distance_goal = self.getDistanceGoal(board)
        spaces_dic = {
            'goal': distance_goal,
            'lava': distance_lava
        }
        next_state = spaces_dic
        self.addToObservations(spaces_dic)

        done = timestep.step_type == 2
        #next_state, reward, done, info = self.env.step(action)
        # modify ...
        return next_state, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        reward = timestep.reward
        board = timestep.observation['board']
        distance_lava = self.getDistanceLava(board,reward)
        distance_goal = self.getDistanceGoal(board)
        spaces_dic = {
            'goal': distance_goal,
            'lava': distance_lava
        }
        next_state = spaces_dic
        self.addToObservations(spaces_dic)

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

    def getDistance(self,board):
        nearestLava = 5
        playerStateX, playerStateY = 2,2
        for x in range(0,5):
            for y in range(0,5):
                if board[x][y] == 4:
                    distance = abs(playerStateX-x) + abs(playerStateY-y)
                    if(distance < nearestLava):
                        nearestLava = distance
        return nearestLava

    def getPlayerState(self,board):
        playerState = (0, 0)
        for i in range(7):
            for j in range(9):
                if board[i][j] == 2:
                    playerState = (i, j)

        return playerState

    def getDistanceLava(self,board,reward):
        nearestLavaState = (6,4)
        playerState = self.getPlayerState(board)
        nearestLavaDistance = 10

        for x in range(7):
            for y in range(9):
                if board[x][y] == 4:
                    distance = abs(playerState[0]-x) + abs(playerState[1]-y)
                    if(distance < nearestLavaDistance):
                        nearestLavaDistance = distance
                        lavaState = (x,y)

        if(reward == -51):
            lavaState = playerState

        return lavaState[0] - playerState[0], lavaState[1] - playerState[1]

    def getDistanceGoal(self,board):
        playerState = self.getPlayerState(board)
        goalState = playerState
        for i in range(7):
            for j in range(9):
                if board[i][j] == 3:
                    goalState = (i, j)

        return  goalState[0] - playerState[0],  goalState[1] - playerState[1]


    def seed(self,seed):
        print(seed)

    def addToObservations(self,observation):
        if observation not in self.boardObservations:
            self.boardObservations.append(observation)

    def getEntities(self,board):

        nearestLava = 5
        playerStateX, playerStateY = 2,2
        entityList = []
        for x in range(0,5):
            for y in range(0,5):
                if board[x][y] == 3:
                    entityList.append()
        return nearestLava

