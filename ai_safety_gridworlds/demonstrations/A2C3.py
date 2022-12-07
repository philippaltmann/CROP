import time
import sys
import pylab
import random
import numpy as np

from collections import deque
from keras.layers import Dense, Conv2D, Reshape
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential

import ai_safety_gridworlds
from ai_safety_gridworlds.helpers import factory

EPISODES = 100000
TEST = False

# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load = False
        self.save_loc = './GridWorld_A2Critic'

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.9
        self.actor_lr  = 0.00001
        self.critic_lr = 0.00003

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load:
            self.load_model()

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()

        # input size: [batch_size, 10, 10, 1]
        # output size: [batch_size, 10, 10, 16]
        actor.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1),
                    activation='relu', padding='same', input_shape=self.state_size,
                    kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 10, 10, 16]
        # output size: [batch_size, 10, 10, 32]
        actor.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
                    activation='relu', padding='same',
                    kernel_initializer='he_uniform'))

        # input size: [batch_size, 10, 10, 32]
        # output size: batch_sizex10x10x32 = 3200xbatch_size
        actor.add(Reshape(target_shape=(1, 3200)))
        actor.add(Dense(3000, activation='relu',
                    kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                    kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        # input size: [batch_size, 10, 10, 1]
        # output size: [batch_size, 10, 10, 16]
        critic.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1),
                    activation='relu', padding='same', input_shape=self.state_size,
                    kernel_initializer='glorot_uniform'))

        # input size: [batch_size, 10, 10, 16]
        # output size: [batch_size, 10, 10, 32]
        critic.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1),
                    activation='relu', padding='same',
                    kernel_initializer='he_uniform'))

        # input size: [batch_size, 10, 10, 32]
        # output size: batch_sizex10x10x32 = 3200xbatch_size
        critic.add(Reshape(target_shape=(1, 3200)))
        critic.add(Dense(3000, activation='relu',
                    kernel_initializer='glorot_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                    kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss='mse', optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * next_value - value
            target[0][0] = reward + self.discount_factor * next_value

        advantages = np.reshape(advantages, (1, advantages.shape[0], advantages.shape[1]))
        target = np.reshape(target, (1, target.shape[0], target.shape[1]))

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    # load the saved model
    def load_model(self):
        self.actor.load_weights(self.save_loc + "_actor.h5")
        self.critic.load_weights(self.save_loc + "_critic.h5")

    # save the model which is under training
    def save_model(self):
        self.actor.save_weights(self.save_loc + "_actor.h5")
        self.critic.save_weights(self.save_loc + "_critic.h5")

    def get_state(self, board):
        for row in range(self.rows):
            for column in range(self.columns):
                if board[row][column] == 2: return np.array([row, column])


if __name__ == "__main__":
    #rad = np.arctan2(-1,2)
    #erg = rad * (180/np.pi);
    #print(erg)
    env = ai_safety_gridworlds.helpers.factory.get_environment_obj('distributional_shift', is_testing=False)
    t = env.reset()
    t = env.step(3)
    t = env.step(1)
    t = env.step(3)
    t = env.step(3)
    t = env.step(3)
    t = env.step(3)
    t = env.step(3)
    #t = env.step(0)
    print(t)
